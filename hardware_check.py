import hashlib
import platform
import subprocess

def get_linux_hardware_info():
    identifiers = []
    
    # Try CPU info
    try:
        cpu_info = subprocess.check_output('cat /proc/cpuinfo | grep "processor"', shell=True).decode()
        identifiers.append(cpu_info)
    except:
        pass
    
    # Try system UUID
    try:
        system_uuid = subprocess.check_output('sudo dmidecode -s system-uuid', shell=True).decode()
        identifiers.append(system_uuid)
    except:
        pass
        
    # Try MAC address of first network interface
    try:
        mac = subprocess.check_output("cat /sys/class/net/$(ls /sys/class/net | head -n 1)/address", shell=True).decode()
        identifiers.append(mac)
    except:
        pass
    
    return "".join(identifiers) if identifiers else "default_linux_id"

def get_hardware_id():
    system = platform.system()
    
    if system == "Windows":
        import wmi
        c = wmi.WMI()
        system_info = c.Win32_ComputerSystemProduct()[0]
        hardware_string = f"{system_info.UUID}{system_info.IdentifyingNumber}"
    
    elif system == "Linux":
        hardware_string = get_linux_hardware_info()
    
    else:
        hardware_string = "unsupported_os"
        
    return hashlib.sha256(hardware_string.encode()).hexdigest()

def verify_license(expected_hash):
    current_id = get_hardware_id()
    return current_id == expected_hash

if __name__ == "__main__":
    print(f"Operating System: {platform.system()}")
    print(f"Hardware ID: {get_hardware_id()}")


