import taichi as ti
import order

ti.init(arch=ti.cpu, debug=True)

# Minimal test 
res = 4
N = res * res

print("Creating fields...")
z_2d = ti.field(ti.f32, shape=(res, res))
z_1d = ti.field(ti.f32, shape=N)
bound_2d = ti.field(ti.i64, shape=(res, res))
bound_1d = ti.field(ti.i64, shape=N)

print("Initializing data...")
# Simple initialization
@ti.kernel
def init_simple():
    for i, j in ti.ndrange(res, res):
        z_2d[i, j] = float(i * res + j + 10)  # Non-zero heights
        bound_2d[i, j] = 1 if (i == 0 or i == res-1 or j == 0 or j == res-1) else 0
        z_1d[i * res + j] = z_2d[i, j]  
        bound_1d[i * res + j] = bound_2d[i, j]

init_simple()

print("Testing RCV computation...")
print(f"z_1d shape: {z_1d.shape}, bound_1d shape: {bound_1d.shape}")
print(f"res: {res}, N: {N}")
print("z_1d values:", [z_1d[i] for i in range(N)])
print("bound_1d values:", [bound_1d[i] for i in range(N)])

rcv_fields = order.declare_rcv_matrix_cuda(res)
rcv, W = rcv_fields
print(f"rcv shape: {rcv.shape}, W shape: {W.shape}")

try:
    rcv, W = order.rcv_matrix(z_1d, bound_1d, rcv_fields)
    print("RCV:", [rcv[i] for i in range(N)])
except Exception as e:
    print("RCV ERROR:", e)

print("Testing drain computation...")
drain = ti.field(ti.f32, shape=N)
for i in range(N):
    drain[i] = 1.0

print("Testing tree accumulation...")
try:
    result = order.tree_accum_upward_(rcv, W, drain)
    print("Tree accum SUCCESS:", [result[i] for i in range(N)])
except Exception as e:
    print("Tree accum ERROR:", e)
    
print("Testing tree downward...")
try:
    tree_fields = order.declare_flow_cuda_tree_accum_downward(N)
    result2 = order.run_flow_cuda_tree_accum_downward(rcv, W, drain, *tree_fields)
    print("Tree down SUCCESS:", [result2[i] for i in range(N)])
except Exception as e:
    print("Tree down ERROR:", e)

print("All tests completed!")