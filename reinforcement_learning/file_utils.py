import time
import platform

PLATFORM_WINDOWS = 'Windows'

if platform.system() == PLATFORM_WINDOWS:
    # conda install -c anaconda pywin32
    import win32file, win32con, pywintypes
else:
    import fcntl


class FileUtils:

    @staticmethod
    def lock_file(f):
        while True:
            try:
                if platform.system() == PLATFORM_WINDOWS:
                    hfile = win32file._get_osfhandle(f.fileno())
                    win32file.LockFileEx(hfile, win32con.LOCKFILE_FAIL_IMMEDIATELY | win32con.LOCKFILE_EXCLUSIVE_LOCK,
                                         0, 0xffff0000, pywintypes.OVERLAPPED())
                else:
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
            except:
                time.sleep(0.1)

    @staticmethod
    def unlock_file(f):
        while True:
            try:
                if platform.system() == PLATFORM_WINDOWS:
                    hfile = win32file._get_osfhandle(f.fileno())
                    win32file.UnlockFileEx(hfile, 0, 0, 0xffff0000, pywintypes.OVERLAPPED())
                else:
                    fcntl.flock(f, fcntl.LOCK_UN)
                break
            except:
                time.sleep(0.1)
