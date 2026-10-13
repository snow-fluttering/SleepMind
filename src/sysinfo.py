"""Hardware, software and memory provenance for reproducible measurements."""
import ctypes
import os
import platform
import subprocess
import sys

import numpy as np
import scipy
import sklearn
import torch


def _cpu_name():
    if os.name == 'nt':
        try:
            out = subprocess.run(['powershell', '-NoProfile', '-Command',
                                  '(Get-CimInstance Win32_Processor).Name'],
                                 capture_output=True, text=True, timeout=20).stdout.strip()
            if out:
                return out.splitlines()[0].strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return platform.processor() or 'unknown'


def _total_ram_bytes():
    if os.name == 'nt':
        class Status(ctypes.Structure):
            _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong),
                        ('total', ctypes.c_ulonglong), ('avail', ctypes.c_ulonglong),
                        ('total_page', ctypes.c_ulonglong), ('avail_page', ctypes.c_ulonglong),
                        ('total_virtual', ctypes.c_ulonglong), ('avail_virtual', ctypes.c_ulonglong),
                        ('avail_ext', ctypes.c_ulonglong)]
        status = Status()
        status.length = ctypes.sizeof(Status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.total)
    return None


def peak_rss_bytes():
    """Peak resident set size of this process (peak working set on Windows)."""
    if os.name == 'nt':
        class Counters(ctypes.Structure):
            _fields_ = [('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong),
                        ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                        ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
                        ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                        ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]
        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(handle), ctypes.byref(counters), counters.cb)
        return int(counters.PeakWorkingSetSize) if ok else None
    import resource
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value * (1 if sys.platform == 'darwin' else 1024))


def environment():
    """Return a JSON-safe description of hardware and library versions."""
    cuda = torch.cuda.is_available()
    return {
        'cpu': _cpu_name(), 'logical_cpus': os.cpu_count(), 'total_ram_bytes': _total_ram_bytes(),
        'platform': platform.platform(), 'python': sys.version.split()[0],
        'torch': torch.__version__, 'numpy': np.__version__, 'scipy': scipy.__version__,
        'scikit_learn': sklearn.__version__, 'torch_threads': torch.get_num_threads(),
        'cuda_available': cuda,
        'cuda_device': torch.cuda.get_device_name(0) if cuda else None,
    }
