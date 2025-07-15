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
        
        # Terrain data
        self.elevation = ti.field(ti.f32, shape=(self.n_cells,))
        self.precipitation = ti.field(ti.f32, shape=(self.n_cells,))
        self.outflow_mask = ti.field(ti.i32, shape=(self.n_cells,))
        
        # Flow routing data
        self.recipients = ti.field(ti.i32, shape=(self.n_cells,))
        self.donors = ti.field(ti.i32, shape=(self.n_cells, 4))
        self.donor_count = ti.field(ti.i32, shape=(self.n_cells,))
        self.discharge = ti.field(ti.f32, shape=(self.n_cells,))
        
        # Ping-pong buffers
        self.discharge_a = ti.field(ti.f32, shape=(self.n_cells,))
        self.discharge_b = ti.field(ti.f32, shape=(self.n_cells,))
        self.recipients_a = ti.field(ti.i32, shape=(self.n_cells,))
        self.recipients_b = ti.field(ti.i32, shape=(self.n_cells,))
        
        # Depression routing data
        self.basin_id = ti.field(ti.i32, shape=(self.n_cells,))
        self.basin_id_a = ti.field(ti.i32, shape=(self.n_cells,))
        self.basin_id_b = ti.field(ti.i32, shape=(self.n_cells,))
        self.local_minima = ti.field(ti.i32, shape=(self.n_cells,))
        self.n_minima = ti.field(ti.i32, shape=())
        self.basin_saddle = ti.field(ti.i32, shape=(self.n_cells,))
        self.basin_outlet = ti.field(ti.i32, shape=(self.n_cells,))
        self.basin_saddle_elevation = ti.field(ti.f32, shape=(self.n_cells,))
        self.basin_valid = ti.field(ti.i32, shape=(self.n_cells,))
        self.border_cells = ti.field(ti.i32, shape=(self.n_cells,))
        self.border_elevation = ti.field(ti.f32, shape=(self.n_cells,))
        self.carving_tag = ti.field(ti.i32, shape=(self.n_cells,))
        self.is_local_minimum = ti.field(ti.i32, shape=(self.n_cells,))

    @ti.func
    def xy_to_index(self, x, y, width):
        return y * width + x

    @ti.func
    def get_neighbors_4(self, idx, width, height):
        x, y = idx % width, idx // width
        neighbors = ti.Vector([-1, -1, -1, -1])
        if x > 0: neighbors[0] = self.xy_to_index(x - 1, y, width)
        if x < width - 1: neighbors[1] = self.xy_to_index(x + 1, y, width)
        if y > 0: neighbors[2] = self.xy_to_index(x, y - 1, width)
        if y < height - 1: neighbors[3] = self.xy_to_index(x, y + 1, width)
        return neighbors

    @ti.kernel
    def compute_recipients(self):
        for idx in range(self.n_cells):
            if self.outflow_mask[idx] == 1:
                self.recipients[idx] = -1
            else:
                neighbors = self.get_neighbors_4(idx, self.width, self.height)
                best_idx = -1
                best_elevation = self.elevation[idx]
                for i in range(4):
                    n_idx = neighbors[i]
                    if n_idx >= 0 and self.elevation[n_idx] < best_elevation:
                        best_elevation = self.elevation[n_idx]
                        best_idx = n_idx
                self.recipients[idx] = best_idx

    @ti.kernel
    def compute_donors(self, rec_in: ti.template()):
        self.donor_count.fill(0)
        self.donors.fill(-1)
        for idx in range(self.n_cells):
            r_idx = rec_in[idx]
            if r_idx >= 0:
                old_count = ti.atomic_add(self.donor_count[r_idx], 1)
                if old_count < 4:
                    self.donors[r_idx, old_count] = idx

    @ti.kernel
    def rake_compress_iteration(self, dis_in: ti.template(), rec_in: ti.template(),
                                dis_out: ti.template(), rec_out: ti.template()):
        for c in range(self.n_cells):
            q_c_new = dis_in[c]
            r_c_new = rec_in[c]
            for i in range(self.donor_count[c]):
                d = self.donors[c, i]
                if d < 0: continue
                if self.donor_count[d] == 0:
                    q_c_new += dis_in[d]
                elif self.donor_count[d] == 1:
                    q_c_new += dis_in[d]
                    r_c_new = rec_in[d]
            dis_out[c] = q_c_new
            rec_out[c] = r_c_new

    def flow_routing(self):
        self.discharge_a.copy_from(self.precipitation)
        self.recipients_a.copy_from(self.recipients)
        dis_in, dis_out = self.discharge_a, self.discharge_b
        rec_in, rec_out = self.recipients_a, self.recipients_b
        max_iter = int(math.log2(self.n_cells)) + 1
        for _ in range(max_iter):
            self.compute_donors(rec_in)
            self.rake_compress_iteration(dis_in, rec_in, dis_out, rec_out)
            dis_in, dis_out = dis_out, dis_in
            rec_in, rec_out = rec_out, rec_in
        self.discharge.copy_from(dis_in)

    @ti.kernel
    def find_local_minima(self):
        count = 0
        for idx in range(self.n_cells):
            if self.recipients[idx] < 0 and self.outflow_mask[idx] == 0:
                count += 1
        self.n_minima[None] = count
        write_idx = 0
        for idx in range(self.n_cells):
            if self.recipients[idx] < 0 and self.outflow_mask[idx] == 0:
                if write_idx < count:
                    self.local_minima[write_idx] = idx
                    write_idx += 1

    def depression_routing(self, use_carving=True):
        self.find_local_minima()
        max_iter = int(math.log2(max(1, self.n_minima[None]))) + 1
        for _ in range(max_iter):
            if self.n_minima[None] == 0: break
            self.propagate_basin_ids()
            self.detect_border_cells()
            self.find_basin_saddles()
            self.remove_cycles()
            if use_carving:
                self.depression_carving()
            else:
                self.depression_jumping()
            self.find_local_minima()

    @ti.kernel
    def _propagate_basin_ids_init(self):
        for idx in range(self.n_cells):
            self.recipients_a[idx] = self.recipients[idx]
            self.basin_id_a[idx] = -1
        for i in range(self.n_minima[None]):
            min_idx = self.local_minima[i]
            self.basin_id_a[min_idx] = i

    @ti.kernel
    def _propagate_basin_ids_iter(self, rec_in: ti.template(), bid_in: ti.template(),
                                  rec_out: ti.template(), bid_out: ti.template()):
        for idx in range(self.n_cells):
            r_idx = rec_in[idx]
            if r_idx >= 0:
                bid_out[idx] = bid_in[r_idx] if bid_in[idx] < 0 else bid_in[idx]
                rr_idx = rec_in[r_idx]
                rec_out[idx] = rr_idx if rr_idx >= 0 else r_idx
            else:
                bid_out[idx] = bid_in[idx]
                rec_out[idx] = rec_in[idx]

    def propagate_basin_ids(self):
        self._propagate_basin_ids_init()
        rec_in, rec_out = self.recipients_a, self.recipients_b
        bid_in, bid_out = self.basin_id_a, self.basin_id_b
        max_iter = int(math.log2(self.n_cells)) + 1
        for _ in range(max_iter):
            self._propagate_basin_ids_iter(rec_in, bid_in, rec_out, bid_out)
            rec_in, rec_out = rec_out, rec_in
            bid_in, bid_out = bid_out, bid_in
        self.basin_id.copy_from(bid_in)

    @ti.kernel
    def detect_border_cells(self):
        for idx in range(self.n_cells):
            self.border_cells[idx] = 0
            self.border_elevation[idx] = 1e9
            current_basin = self.basin_id[idx]
            is_border = 0
            min_neighbor_elevation = 1e9
            for i in range(4):
                n_idx = self.get_neighbors_4(idx, self.width, self.height)[i]
                if n_idx >= 0:
                    if self.basin_id[n_idx] != current_basin:
                        is_border = 1
                    min_neighbor_elevation = ti.min(min_neighbor_elevation, self.elevation[n_idx])
            if is_border:
                self.border_cells[idx] = 1
                self.border_elevation[idx] = ti.max(self.elevation[idx], min_neighbor_elevation)

    @ti.kernel
    def find_basin_saddles(self):
        self.basin_saddle.fill(-1)
        self.basin_outlet.fill(-1)
        self.basin_saddle_elevation.fill(1e9)
        self.basin_valid.fill(0)
        for idx in range(self.n_cells):
            if self.border_cells[idx] == 1:
                current_basin = self.basin_id[idx]
                if current_basin < self.n_minima[None]:
                    if self.border_elevation[idx] < self.basin_saddle_elevation[current_basin]:
                        self.basin_saddle_elevation[current_basin] = self.border_elevation[idx]
                        self.basin_saddle[current_basin] = idx
                        best_outlet, best_outlet_elevation = -1, 1e9
                        for i in range(4):
                            n_idx = self.get_neighbors_4(idx, self.width, self.height)[i]
                            if n_idx >= 0 and self.basin_id[n_idx] != current_basin and self.elevation[n_idx] < best_outlet_elevation:
                                best_outlet_elevation = self.elevation[n_idx]
                                best_outlet = n_idx
                        self.basin_outlet[current_basin] = best_outlet
                        self.basin_valid[current_basin] = 1 if best_outlet >= 0 else 0

    @ti.kernel
    def remove_cycles(self):
        for i in range(self.n_minima[None]):
            if self.basin_valid[i] == 1:
                outlet_idx = self.basin_outlet[i]
                if outlet_idx >= 0:
                    outlet_basin = self.basin_id[outlet_idx]
                    if outlet_basin < self.n_minima[None] and self.basin_valid[outlet_basin] == 1:
                        their_outlet = self.basin_outlet[outlet_basin]
                        if their_outlet >= 0 and self.basin_id[their_outlet] == i:
                            if i > outlet_basin:
                                self.basin_valid[i] = 0

    def depression_carving(self):
        self._depression_carving_tag()
        self._depression_carving_propagate()
        self.is_local_minimum.fill(0)
        for i in range(self.n_minima[None]):
            self.is_local_minimum[self.local_minima[i]] = 1
        self._depression_carving_reverse()
        self._depression_carving_set_outlets()

    @ti.kernel
    def _depression_carving_tag(self):
        self.carving_tag.fill(0)
        for i in range(self.n_minima[None]):
            if self.basin_valid[i] == 1:
                saddle_idx = self.basin_saddle[i]
                if saddle_idx >= 0:
                    self.carving_tag[saddle_idx] = 1

    @ti.kernel
    def _depression_carving_propagate(self):
        max_iter = ti.cast(ti.math.log2(ti.cast(self.n_cells, ti.f32)), ti.i32) + 1
        for _ in range(max_iter):
            for idx in range(self.n_cells):
                if self.carving_tag[idx] == 1:
                    r_idx = self.recipients[idx]
                    if r_idx >= 0:
                        self.carving_tag[r_idx] = 1
                        rr_idx = self.recipients[r_idx]
                        if rr_idx >= 0:
                            self.recipients[idx] = rr_idx

    @ti.kernel
    def _depression_carving_reverse(self):
        for idx in range(self.n_cells):
            r_idx = self.recipients[idx]
            if r_idx >= 0 and self.carving_tag[r_idx] == 1 and self.is_local_minimum[idx] == 0:
                self.recipients[r_idx] = idx

    @ti.kernel
    def _depression_carving_set_outlets(self):
        for i in range(self.n_minima[None]):
            if self.basin_valid[i] == 1:
                saddle_idx = self.basin_saddle[i]
                outlet_idx = self.basin_outlet[i]
                if saddle_idx >= 0 and outlet_idx >= 0:
                    self.recipients[saddle_idx] = outlet_idx

    @ti.kernel
    def depression_jumping(self):
        for i in range(self.n_minima[None]):
            if self.basin_valid[i] == 1:
                min_idx = self.local_minima[i]
                outlet_idx = self.basin_outlet[i]
                if outlet_idx >= 0:
                    self.recipients[min_idx] = outlet_idx

    def run_fastflow(self, elevation_data, precipitation_data, outflow_data, depression=False):
        self.elevation.from_numpy(elevation_data.flatten())
        self.precipitation.from_numpy(precipitation_data.flatten())
        self.outflow_mask.from_numpy(outflow_data.flatten())
        self.compute_recipients()
        if depression:
            self.depression_routing(True)
        self.flow_routing()
        return self.discharge.to_numpy().reshape((self.height, self.width)), self.recipients.to_numpy()

if __name__ == "__main__":
    width, height = 512, 512
    x = np.linspace(-2, 2, width)
    y = np.linspace(-2, 2, height)
    X, Y = np.meshgrid(x, y)
    elevation = 100 * np.exp(-(X**2 + Y**2) / 2)
    center_x, center_y = width // 2, height // 2
    depression_radius = 50
    for dy in range(-depression_radius, depression_radius):
        for dx in range(-depression_radius, depression_radius):
            if dx*dx + dy*dy < depression_radius*depression_radius:
                px, py = center_x + dx, center_y + dy
                if 0 <= px < width and 0 <= py < height:
                    elevation[py, px] -= 20
    precipitation = np.ones((height, width))
    outflow = np.zeros((height, width), dtype=np.int32)
    outflow[0, :] = 1
    outflow[-1, :] = 1
    outflow[:, 0] = 1
    outflow[:, -1] = 1
    
    fastflow = FastFlow(width, height)
    discharge, recipients = fastflow.run_fastflow(elevation, precipitation, outflow, depression=True)

    import matplotlib.pyplot as plt
    plt.imshow(np.log10(discharge.clip(min=1)), cmap="Blues")
    plt.title("Log of Discharge")
    plt.colorbar()
    plt.show()
    
    print(f"Computed discharge range: {discharge.min():.2f} to {discharge.max():.2f}")
    print(f"Mean discharge: {discharge.mean():.2f}")
    print("FastFlow computation completed successfully!")