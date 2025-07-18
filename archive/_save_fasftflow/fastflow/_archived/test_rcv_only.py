import taichi as ti
import order

ti.init(arch=ti.cpu, debug=True)

# Test just RCV computation 
res = 4
N = res * res

z_1d = ti.field(ti.f32, shape=N)
bound_1d = ti.field(ti.i64, shape=N)

@ti.kernel
def init_test():
    for i in range(N):
        x = i % res
        y = i // res
        z_1d[i] = float(10 + i)  # Simple increasing heights
        bound_1d[i] = 1 if (x == 0 or x == res-1 or y == 0 or y == res-1) else 0

init_test()

print("Heights (1D):", [z_1d[i] for i in range(N)])
print("Boundaries (1D):", [bound_1d[i] for i in range(N)])

# Test RCV computation
rcv_fields = order.declare_rcv_matrix_cuda(res)
rcv, W = order.rcv_matrix(z_1d, bound_1d, rcv_fields)

print("RCV:", [rcv[i] for i in range(N)])
print("W:", [W[i] for i in range(N)])

# Show in 2D format for easier understanding
print("\n2D view:")
print("Heights:")
for y in range(res):
    row = [f"{z_1d[y*res + x]:5.1f}" for x in range(res)]
    print("  " + " ".join(row))

print("Receivers:")
for y in range(res):
    row = []
    for x in range(res):
        idx = y * res + x
        rcv_val = rcv[idx]
        if rcv_val == idx:
            row.append("  X")
        else:
            rcv_x, rcv_y = rcv_val % res, rcv_val // res
            row.append(f"{rcv_x},{rcv_y}")
    print("  " + " ".join(row))