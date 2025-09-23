#!/home/ver/env2/bin/python


import pexpect
import time
import sys
import subprocess

def get_bt_info(mac):
    result = subprocess.run(["bluetoothctl", "info", mac],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True)
   
    output = result.stdout
    return { "paired": "Paired: yes" in output,
        "trusted": "Trusted: yes" in output,
        "connected": "Connected: yes" in output}


def pair_trust_connect(mac_address):
    print(f"Starting pairing process for {mac_address}")
    child = pexpect.spawn("bluetoothctl", echo=False)
   
    info = get_bt_info(mac_address)
   
    trusted = info['trusted']
    paired = info['paired']
    connected = info['connected']
   
    if connected:
        print("Connected already")
        return
    elif trusted and paired:
        print("Paired and trusted, connecting")
        for _ in range(3):
            child.sendline(f"connect {mac_address}")
            time.sleep(2)
            index = child.expect(["Connection successful", "Failed to connect", pexpect.TIMEOUT], timeout=20)
           
            if index == 0:
                print("✅ Connected")
                return
            else:
                print("⚠️ Connection may have failed (check device is on)")
   
    # Power on Bluetooth
    child.sendline("power on")
    child.sendline("default-agent")

    # Start scanning (optional, can help discovery)
    child.sendline("scan on")
    time.sleep(10)
    child.sendline("scan off")
   
    # Pair
    child.sendline(f"pair {mac_address}")
    index = child.expect(["Pairing successful", "Failed to pair", "AuthenticationFailed", pexpect.TIMEOUT], timeout=30)
    if index != 0:
        print("❌ Pairing failed")
        child.sendline("exit")
        return False
       
    print("✅ Paired successfully")
    # Trust
    child.sendline(f"trust {mac_address}")
    print("✅ Trusted device")
    time.sleep(2)
    # Connect
    for _ in range(3):
        child.sendline(f"connect {mac_address}")
        time.sleep(2)
        index = child.expect(["Connection successful", "Failed to connect", pexpect.TIMEOUT], timeout=20)
       
        if index == 0:
            print("✅ Connected")
            break
        else:
            print("⚠️ Connection may have failed (check device is on)")

    child.sendline("exit")
    return True

# Example usage:
if __name__ == "__main__":
    mac = ""  # Replace with your headphone's MAC address
    pair_trust_connect(mac)
