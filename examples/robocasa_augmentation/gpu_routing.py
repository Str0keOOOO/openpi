"""Identify an EGL device by its binary UUID, independent of CUDA/NVML ordinals."""
import ctypes as c
import uuid


def egl_device_for_uuid(gpu_uuid):
    target = uuid.UUID(gpu_uuid.removeprefix('GPU-')).bytes
    lib = c.CDLL('libEGL.so.1')
    lib.eglGetProcAddress.restype = c.c_void_p
    lib.eglGetProcAddress.argtypes = [c.c_char_p]
    query_ptr = lib.eglGetProcAddress(b'eglQueryDevicesEXT')
    binary_ptr = lib.eglGetProcAddress(b'eglQueryDeviceBinaryEXT')
    if not query_ptr or not binary_ptr:
        raise RuntimeError('EGL device enumeration/UUID extension unavailable')
    query = c.CFUNCTYPE(c.c_uint, c.c_int, c.POINTER(c.c_void_p), c.POINTER(c.c_int))(query_ptr)
    binary = c.CFUNCTYPE(c.c_uint, c.c_void_p, c.c_int, c.c_int, c.c_void_p, c.POINTER(c.c_int))(binary_ptr)
    devices = (c.c_void_p * 32)()
    count = c.c_int()
    if not query(32, devices, c.byref(count)):
        raise RuntimeError('EGL device enumeration failed')
    for index in range(count.value):
        data = (c.c_ubyte * 16)()
        size = c.c_int()
        if binary(devices[index], 0x335C, 16, data, c.byref(size)) and size.value == 16 and bytes(data) == target:
            return index
    raise RuntimeError(f'Requested GPU UUID is not present in EGL devices: {gpu_uuid}')
