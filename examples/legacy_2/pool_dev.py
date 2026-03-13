import taichi as ti

import pyfastflow as pf


def yolo(nx):
    print("FUNC")
    f2 = pf.pool.taipool.get_tpfield(dtype=ti.f32, shape=(nx))
    print(f2)
    print(pf.pool.taipool.stats())
    print("ENDFUNC")


ti.init(ti.gpu)

nx = 512
ny = 512

# print(pf.pool.taipool)

f1 = pf.pool.taipool.get_tpfield(dtype=ti.i32, shape=(nx * ny))
f1 = pf.pool.taipool.get_tpfield(dtype=ti.f32, shape=(nx * ny))
f1 = pf.pool.taipool.get_tpfield(dtype=ti.i32, shape=(nx * ny))
f1.release()
f1 = pf.pool.taipool.get_tpfield(dtype=ti.i8, shape=(nx * ny))
print(f1)
print(pf.pool.taipool.stats())
yolo(nx)
print(pf.pool.taipool.stats())
