import taichi as ti
import numpy as np
import math

ti.init(arch=ti.gpu)

@ti.data_oriented
class FastFlow:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.n_cells = width * height
        
        # Terrain data (row-major vectorized)
        self.elevation = ti.field(ti.f32, shape=(self.n_cells,))
        self.precipitation = ti.field(ti.f32, shape=(self.n_cells,))
        self.outflow_mask = ti.field(ti.i32, shape=(self.n_cells,))
        
        # Flow routing data
        self.recipients = ti.field(ti.i32, shape=(self.n_cells,))  # recipient cell index
        self.donors = ti.field(ti.i32, shape=(self.n_cells, 4))   # up to 4 donor indices per cell
        self.donor_count = ti.field(ti.i32, shape=(self.n_cells,))
        self.discharge = ti.field(ti.f32, shape=(self.n_cells,))
        
        # Ping-pong buffers for rake-compress (Algorithm 1)
        self.discharge_a = ti.field(ti.f32, shape=(self.n_cells,))
        self.discharge_b = ti.field(ti.f32, shape=(self.n_cells,))
        self.recipients_a = ti.field(ti.i32, shape=(self.n_cells,))
        self.recipients_b = ti.field(ti.i32, shape=(self.n_cells,))
        self.source_buffer = ti.field(ti.i32, shape=(self.n_cells,))  # ping-pong state
        
        # Depression routing data (Algorithm 2-5)
        self.basin_id = ti.field(ti.i32, shape=(self.n_cells,))
        self.local_minima = ti.field(ti.i32, shape=(self.n_cells,))  # list of local minima indices
        self.n_minima = ti.field(ti.i32, shape=())
        
        # Basin data for depression graph
        self.basin_saddle = ti.field(ti.i32, shape=(self.n_cells,))      # saddle cell index
        self.basin_outlet = ti.field(ti.i32, shape=(self.n_cells,))      # outlet cell index  
        self.basin_saddle_elevation = ti.field(ti.f32, shape=(self.n_cells,))
        self.basin_valid = ti.field(ti.i32, shape=(self.n_cells,))
        
        # Border detection (Algorithm 3)
        self.border_cells = ti.field(ti.i32, shape=(self.n_cells,))
        self.border_elevation = ti.field(ti.f32, shape=(self.n_cells,))
        
        # Depression carving (Algorithm 4)
        self.carving_tag = ti.field(ti.i32, shape=(self.n_cells,))
        
    @ti.func
    def xy_to_index(self, x: ti.i32, y: ti.i32, width: ti.i32) -> ti.i32:
        """Convert 2D coordinates to 1D index (row-major)"""
        return y * width + x
        
    @ti.func
    def index_to_xy(self, idx: ti.i32, width: ti.i32) -> ti.types.vector(2, ti.i32):
        """Convert 1D index to 2D coordinates (row-major)"""
        return ti.Vector([idx % width, idx // width])
        
    @ti.func
    def get_neighbors_4(self, idx: ti.i32, width: ti.i32, height: ti.i32) -> ti.types.vector(4, ti.i32):
        """Get 4-connected neighbor indices"""
        xy = self.index_to_xy(idx, width)
        x, y = xy[0], xy[1]
        neighbors = ti.Vector([-1, -1, -1, -1])
        
        # Left neighbor
        if x > 0:
            neighbors[0] = self.xy_to_index(x - 1, y, width)
        # Right neighbor  
        if x < width - 1:
            neighbors[1] = self.xy_to_index(x + 1, y, width)
        # Up neighbor
        if y > 0:
            neighbors[2] = self.xy_to_index(x, y - 1, width)
        # Down neighbor
        if y < height - 1:
            neighbors[3] = self.xy_to_index(x, y + 1, width)
            
        return neighbors

    @ti.kernel
    def compute_recipients(self, elevation: ti.template(), recipients: ti.template(), 
                          outflow_mask: ti.template(), width: ti.i32, height: ti.i32):
        """Compute recipients using steepest descent (SFD)"""
        for idx in range(width * height):
            if outflow_mask[idx] == 1:
                recipients[idx] = -1  # Outflow cells have no recipients
            else:
                neighbors = self.get_neighbors_4(idx, width, height)
                best_idx = -1
                best_elevation = elevation[idx]
                
                # Find steepest descent neighbor
                for i in range(4):
                    n_idx = neighbors[i]
                    if n_idx >= 0 and elevation[n_idx] < best_elevation:
                        best_elevation = elevation[n_idx]
                        best_idx = n_idx
                
                recipients[idx] = best_idx

    @ti.kernel  
    def compute_donors(self, recipients: ti.template(), donors: ti.template(), 
                      donor_count: ti.template(), n_cells: ti.i32):
        """Compute donors from recipients (used in flow routing)"""
        # Clear donor counts
        for idx in range(n_cells):
            donor_count[idx] = 0
            for i in range(4):
                donors[idx, i] = -1
        
        # Build donor relationships
        for idx in range(n_cells):
            r_idx = recipients[idx]
            if r_idx >= 0:  # Valid recipient
                old_count = ti.atomic_add(donor_count[r_idx], 1)
                if old_count < 4:
                    donors[r_idx, old_count] = idx

    @ti.kernel
    def initialize_flow_routing(self, precipitation: ti.template(), discharge_a: ti.template(),
                              discharge_b: ti.template(), recipients_a: ti.template(),
                              recipients_b: ti.template(), source_buffer: ti.template(),
                              recipients: ti.template(), n_cells: ti.i32):
        """Initialize flow routing ping-pong buffers"""
        for idx in range(n_cells):
            discharge_a[idx] = precipitation[idx]
            discharge_b[idx] = precipitation[idx]
            recipients_a[idx] = recipients[idx]
            recipients_b[idx] = recipients[idx]
            source_buffer[idx] = 0  # Start with buffer A (sign bit = 0)

    @ti.kernel
    def rake_compress_iteration(self, discharge_a: ti.template(), discharge_b: ti.template(),
                               recipients_a: ti.template(), recipients_b: ti.template(),
                               source_buffer: ti.template(), donors: ti.template(),
                               donor_count: ti.template(), n_cells: ti.i32, iteration: ti.i32):
        """Single iteration of rake-compress algorithm (Algorithm 1)"""
        for idx in range(n_cells):
            # Determine read/write buffers based on ping-pong state
            sign_bit = (source_buffer[idx] >> 31) & 1
            last_iter = source_buffer[idx] & 0x7FFFFFFF
            
            # Determine which buffer to read from
            # If sign_bit = 0, data is in A; if sign_bit = 1, data is in B
            read_from_a = 1 if sign_bit == 0 else 0
            
            # If already updated this iteration, skip
            if last_iter == iteration:
                continue
            
            # Read current state
            current_discharge = discharge_a[idx] if read_from_a == 1 else discharge_b[idx]
            current_recipient = recipients_a[idx] if read_from_a == 1 else recipients_b[idx]
            
            # Skip if no valid recipient
            if current_recipient < 0:
                continue
            
            # Process donors (rake and compress operations)
            updated = 0
            new_discharge = current_discharge
            new_recipient = current_recipient

            # Status: 0 don't care, 1: leaf, 2: d has a single donor
            status = 0
            
            for i in range(donor_count[idx]):
                donor_idx = donors[idx, i]
                if donor_idx >= 0:
                    # Read donor's state from appropriate buffer
                    donor_sign = (source_buffer[donor_idx] >> 31) & 1
                    donor_iter = source_buffer[donor_idx] & 0x7FFFFFFF
                    
                    donor_read_a = 1 if donor_sign == 0 else 0
                    
                    donor_discharge = discharge_a[donor_idx] if donor_read_a == 1 else discharge_b[donor_idx]
                    donor_recipient = recipients_a[donor_idx] if donor_read_a == 1 else recipients_b[donor_idx]
                    donor_count_val = donor_count[donor_idx]
                    
                    # Rake: if donor is leaf (no donors)
                    if donor_count_val == 0:
                        new_discharge += donor_discharge
                        updated = 1
                        status = 1
                    
                    # Compress: if donor has single donor
                    elif donor_count_val == 1:
                        new_discharge += donor_discharge
                        new_recipient = donor_recipient
                        updated = 1
                        status = 2
            
            # Write updated state if modified
            if updated == 1:
                # Write to opposite buffer
                new_sign = 1
                if read_from_a == 1:
                    # Was in A, write to B
                    discharge_b[idx] = new_discharge
                    recipients_b[idx] = new_recipient
                else:
                    # Was in B, write to A
                    discharge_a[idx] = new_discharge
                    recipients_a[idx] = new_recipient
                    new_sign = 0
                
                # Update source buffer
                source_buffer[idx] = (new_sign << 31) | iteration

    @ti.kernel
    def update_donor_counts(self, recipients_a: ti.template(), recipients_b: ti.template(),
                           source_buffer: ti.template(), donors: ti.template(),
                           donor_count: ti.template(), n_cells: ti.i32):
        """Update donor counts after rake-compress iteration"""
        # Clear counts
        for idx in range(n_cells):
            donor_count[idx] = 0
        
        # Recount based on current recipients
        for idx in range(n_cells):
            sign_bit = (source_buffer[idx] >> 31) & 1
            read_from_a = 1 if sign_bit == 0 else 0
            
            recipient = recipients_a[idx] if read_from_a == 1 else recipients_b[idx]
            
            if recipient >= 0:
                ti.atomic_add(donor_count[recipient], 1)

    @ti.kernel
    def finalize_flow_routing(self, discharge_a: ti.template(), discharge_b: ti.template(),
                             discharge: ti.template(), source_buffer: ti.template(), n_cells: ti.i32):
        """Merge ping-pong buffers to final discharge"""
        for idx in range(n_cells):
            sign_bit = (source_buffer[idx] >> 31) & 1
            
            # Use the buffer indicated by the sign bit
            if sign_bit == 0:
                discharge[idx] = discharge_a[idx]
            else:
                discharge[idx] = discharge_b[idx]

    @ti.kernel
    def find_local_minima(self, recipients: ti.template(), outflow_mask: ti.template(),
                         local_minima: ti.template(), n_minima: ti.template(), n_cells: ti.i32):
        """Find local minima (cells without valid recipients, excluding outflow)"""
        # Count local minima first
        count = 0
        for idx in range(n_cells):
            if recipients[idx] < 0 and outflow_mask[idx] == 0:
                count += 1
        
        n_minima[None] = count
        
        # Collect local minima indices
        write_idx = 0
        for idx in range(n_cells):
            if recipients[idx] < 0 and outflow_mask[idx] == 0:
                if write_idx < count:
                    local_minima[write_idx] = idx
                    write_idx += 1

    @ti.kernel
    def propagate_basin_ids(self, recipients: ti.template(), basin_id: ti.template(),
                           local_minima: ti.template(), n_minima_val: ti.i32, n_cells: ti.i32):
        """Propagate basin identifiers using pointer jumping (Algorithm 2)"""
        # Initialize basin IDs for local minima
        for i in range(n_minima_val):
            min_idx = local_minima[i]
            basin_id[min_idx] = i
        
        # Pointer jumping to propagate basin IDs
        max_iterations = ti.cast(ti.log2(ti.cast(n_cells, ti.f32)), ti.i32) + 1
        for iter in range(max_iterations):
            for idx in range(n_cells):
                r_idx = recipients[idx]
                if r_idx >= 0:
                    basin_id[idx] = basin_id[r_idx]
                    # Jump recipients for compression
                    rr_idx = recipients[r_idx]
                    if rr_idx >= 0:
                        recipients[idx] = rr_idx

    @ti.kernel
    def detect_border_cells(self, basin_id: ti.template(), elevation: ti.template(),
                           border_cells: ti.template(), border_elevation: ti.template(),
                           width: ti.i32, height: ti.i32):
        """Detect border cells between basins (Algorithm 3)"""
        for idx in range(width * height):
            border_cells[idx] = 0
            border_elevation[idx] = 1e9
            
            neighbors = self.get_neighbors_4(idx, width, height)
            current_basin = basin_id[idx]
            
            min_neighbor_elevation = 1e9
            is_border = 0
            
            for i in range(4):
                n_idx = neighbors[i]
                if n_idx >= 0:
                    neighbor_basin = basin_id[n_idx]
                    if neighbor_basin != current_basin:
                        is_border = 1
                    min_neighbor_elevation = ti.min(min_neighbor_elevation, elevation[n_idx])
            
            if is_border == 1:
                border_cells[idx] = 1
                border_elevation[idx] = ti.max(elevation[idx], min_neighbor_elevation)

    @ti.kernel
    def find_basin_saddles(self, basin_id: ti.template(), border_cells: ti.template(),
                          border_elevation: ti.template(), elevation: ti.template(),
                          basin_saddle: ti.template(), basin_outlet: ti.template(),
                          basin_saddle_elevation: ti.template(), basin_valid: ti.template(),
                          local_minima: ti.template(), n_minima_val: ti.i32,
                          width: ti.i32, height: ti.i32):
        """Find saddles and outlets for each basin (Algorithm 3)"""
        # Initialize basin data
        for i in range(n_minima_val):
            basin_saddle[i] = -1
            basin_outlet[i] = -1
            basin_saddle_elevation[i] = 1e9
            basin_valid[i] = 0
        
        # Find minimum saddle for each basin
        for idx in range(width * height):
            if border_cells[idx] == 1:
                current_basin = basin_id[idx]
                if current_basin < n_minima_val:
                    # Atomic comparison for minimum saddle elevation
                    if border_elevation[idx] < basin_saddle_elevation[current_basin]:
                        basin_saddle_elevation[current_basin] = border_elevation[idx]
                        basin_saddle[current_basin] = idx
                        
                        # Find outlet (lowest neighbor in different basin)
                        neighbors = self.get_neighbors_4(idx, width, height)
                        best_outlet = -1
                        best_outlet_elevation = 1e9
                        
                        for i in range(4):
                            n_idx = neighbors[i]
                            if n_idx >= 0:
                                neighbor_basin = basin_id[n_idx]
                                if neighbor_basin != current_basin and elevation[n_idx] < best_outlet_elevation:
                                    best_outlet_elevation = elevation[n_idx]
                                    best_outlet = n_idx
                        
                        basin_outlet[current_basin] = best_outlet
                        basin_valid[current_basin] = 1 if best_outlet >= 0 else 0

    @ti.kernel
    def remove_cycles(self, basin_outlet: ti.template(), basin_saddle: ti.template(),
                     basin_valid: ti.template(), basin_id: ti.template(),
                     n_minima_val: ti.i32):
        """Remove cycles in depression graph (Algorithm 3)"""
        for i in range(n_minima_val):
            if basin_valid[i] == 1:
                outlet_idx = basin_outlet[i]
                if outlet_idx >= 0:
                    outlet_basin = basin_id[outlet_idx]
                    if outlet_basin < n_minima_val and basin_valid[outlet_basin] == 1:
                        # Check for cycle
                        their_outlet = basin_outlet[outlet_basin]
                        if their_outlet >= 0 and basin_id[their_outlet] == i:
                            # Cycle detected, remove edge with higher basin ID
                            if i > outlet_basin:
                                basin_valid[i] = 0

    @ti.kernel
    def depression_jumping(self, local_minima: ti.template(), basin_outlet: ti.template(),
                          basin_valid: ti.template(), recipients: ti.template(),
                          n_minima_val: ti.i32):
        """Depression jumping variant (Algorithm 4)"""
        for i in range(n_minima_val):
            if basin_valid[i] == 1:
                min_idx = local_minima[i]
                outlet_idx = basin_outlet[i]
                if outlet_idx >= 0:
                    recipients[min_idx] = outlet_idx

    @ti.kernel  
    def depression_carving_tag(self, basin_saddle: ti.template(), basin_valid: ti.template(),
                              carving_tag: ti.template(), n_minima_val: ti.i32, n_cells: ti.i32):
        """Tag saddle cells for depression carving (Algorithm 4)"""
        for idx in range(n_cells):
            carving_tag[idx] = 0
            
        for i in range(n_minima_val):
            if basin_valid[i] == 1:
                saddle_idx = basin_saddle[i]
                if saddle_idx >= 0:
                    carving_tag[saddle_idx] = 1

    @ti.kernel
    def depression_carving_propagate(self, recipients: ti.template(), carving_tag: ti.template(),
                                    n_cells: ti.i32):
        """Propagate carving tags (Algorithm 4)"""
        max_iterations = ti.cast(ti.log2(ti.cast(n_cells, ti.f32)), ti.i32) + 1
        for iter in range(max_iterations):
            for idx in range(n_cells):
                if carving_tag[idx] == 1:
                    r_idx = recipients[idx]
                    if r_idx >= 0:
                        carving_tag[r_idx] = 1
                        # Pointer jumping
                        rr_idx = recipients[r_idx]
                        if rr_idx >= 0:
                            recipients[idx] = rr_idx

    @ti.kernel
    def depression_carving_reverse(self, recipients: ti.template(), carving_tag: ti.template(),
                                  local_minima: ti.template(), n_minima_val: ti.i32, n_cells: ti.i32):
        """Reverse flow paths and set outlets (Algorithm 4)"""
        # Reverse tagged paths (except local minima)
        for idx in range(n_cells):
            if carving_tag[idx] == 1:
                r_idx = recipients[idx]
                if r_idx >= 0 and carving_tag[r_idx] == 1:
                    # Check if recipient is not a local minimum
                    is_local_min = 0
                    for i in range(n_minima_val):
                        if local_minima[i] == r_idx:
                            is_local_min = 1
                            break
                    
                    if is_local_min == 0:
                        recipients[r_idx] = idx

    @ti.kernel
    def depression_carving_set_outlets(self, basin_saddle: ti.template(), basin_outlet: ti.template(),
                                      basin_valid: ti.template(), recipients: ti.template(),
                                      n_minima_val: ti.i32):
        """Set saddle to outlet connections (Algorithm 4)"""
        for i in range(n_minima_val):
            if basin_valid[i] == 1:
                saddle_idx = basin_saddle[i]
                outlet_idx = basin_outlet[i]
                if saddle_idx >= 0 and outlet_idx >= 0:
                    recipients[saddle_idx] = outlet_idx

    def flow_routing(self):
        """Execute complete flow routing using rake-compress (Algorithm 1)"""
        # Compute recipients and donors
        self.compute_recipients(self.elevation, self.recipients, self.outflow_mask,
                               self.width, self.height)
        self.compute_donors(self.recipients, self.donors, self.donor_count, self.n_cells)
        
        # Initialize ping-pong buffers
        self.initialize_flow_routing(self.precipitation, self.discharge_a, self.discharge_b,
                                   self.recipients_a, self.recipients_b, self.source_buffer,
                                   self.recipients, self.n_cells)
        
        # Rake-compress iterations
        max_iterations = int(math.log2(self.n_cells)) + 1
        for iteration in range(max_iterations):
            self.rake_compress_iteration(self.discharge_a, self.discharge_b,
                                       self.recipients_a, self.recipients_b,
                                       self.source_buffer, self.donors, self.donor_count,
                                       self.n_cells, iteration + 1)  # Start from iteration 1
            
            # Update donor counts after each iteration
            self.update_donor_counts(self.recipients_a, self.recipients_b,
                                   self.source_buffer, self.donors, self.donor_count,
                                   self.n_cells)
        
        # Finalize results
        self.finalize_flow_routing(self.discharge_a, self.discharge_b, self.discharge,
                                 self.source_buffer, self.n_cells)

    def depression_routing(self, use_carving=True):
        """Execute complete depression routing (Algorithms 2-5)"""
        max_iterations = int(math.log2(max(self.n_cells, 2))) + 1
        
        for iteration in range(max_iterations):
            # Find local minima
            self.find_local_minima(self.recipients, self.outflow_mask, 
                                 self.local_minima, self.n_minima, self.n_cells)
            
            if self.n_minima[None] == 0:
                break
                
            # Propagate basin IDs
            self.propagate_basin_ids(self.recipients, self.basin_id, self.local_minima,
                                   self.n_minima[None], self.n_cells)
            
            # Find saddles and outlets
            self.detect_border_cells(self.basin_id, self.elevation, self.border_cells,
                                   self.border_elevation, self.width, self.height)
            
            self.find_basin_saddles(self.basin_id, self.border_cells, self.border_elevation,
                                  self.elevation, self.basin_saddle, self.basin_outlet,
                                  self.basin_saddle_elevation, self.basin_valid,
                                  self.local_minima, self.n_minima[None], self.width, self.height)
            
            # Remove cycles
            self.remove_cycles(self.basin_outlet, self.basin_saddle, self.basin_valid,
                             self.basin_id, self.n_minima[None])
            
            # Reroute recipients
            if use_carving:
                self.depression_carving_tag(self.basin_saddle, self.basin_valid,
                                          self.carving_tag, self.n_minima[None], self.n_cells)
                self.depression_carving_propagate(self.recipients, self.carving_tag, self.n_cells)
                self.depression_carving_reverse(self.recipients, self.carving_tag,
                                              self.local_minima, self.n_minima[None], self.n_cells)
                self.depression_carving_set_outlets(self.basin_saddle, self.basin_outlet,
                                                  self.basin_valid, self.recipients, self.n_minima[None])
            else:
                self.depression_jumping(self.local_minima, self.basin_outlet, self.basin_valid,
                                      self.recipients, self.n_minima[None])

    def run_fastflow(self, elevation_data, precipitation_data, outflow_data, depression=True):
        """Run complete FastFlow algorithm"""
        # Set input data
        self.elevation.from_numpy(elevation_data.flatten())
        self.precipitation.from_numpy(precipitation_data.flatten())
        self.outflow_mask.from_numpy(outflow_data.flatten())
        
        # Execute depression routing first if requested
        if depression:
            self.depression_routing(True)
        
        # Then execute flow routing
        self.flow_routing()
        
        # Return results
        discharge_result = self.discharge.to_numpy().reshape((self.height, self.width))
        recipients_result = self.recipients.to_numpy()
        
        return discharge_result, recipients_result


# Example usage
if __name__ == "__main__":
    # Create test terrain
    width, height = 512, 512
    
    import dg2

    print("Generating terrain...")
    noise = dg2.PerlinNoiseF32(frequency=0.01, amplitude=1.0, octaves=6)
    elevation = noise.create_noise_grid(width, height, 0, 0, 100, 100).as_numpy()
    elevation = (elevation - elevation.min()) / (elevation.max() - elevation.min()) * 1000
    
    # Uniform precipitation
    precipitation = np.ones((height, width))
    
    # Boundary outflow
    outflow = np.zeros((height, width), dtype=np.int32)
    outflow[0, :] = 1  # Top edge
    outflow[-1, :] = 1  # Bottom edge
    outflow[:, 0] = 1   # Left edge
    outflow[:, -1] = 1  # Right edge
    
    # Run FastFlow
    fastflow = FastFlow(width, height)
    discharge, recipients = fastflow.run_fastflow(elevation, precipitation, outflow, depression=False)

    import matplotlib.pyplot as plt

    plt.imshow(np.log10(discharge + 1), cmap="Blues")
    plt.colorbar(label="log10(discharge + 1)")
    plt.title("Water Discharge (log scale)")
    plt.show()
    
    print(f"Computed discharge range: {discharge.min():.2f} to {discharge.max():.2f}")
    print(f"Mean discharge: {discharge.mean():.2f}")
    print("FastFlow computation completed successfully!")