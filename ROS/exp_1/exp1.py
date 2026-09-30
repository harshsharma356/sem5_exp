import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def box(ax, x, y, w, h, text, fontsize=9):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.03,rounding_size=0.08",
        linewidth=1.5,
        edgecolor="black",
        facecolor="white"
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize
    )

def layer(ax, x, y, w, h, text):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        linewidth=1.8,
        edgecolor="black",
        facecolor="white"
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold"
    )

def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=14,
            linewidth=1.4,
            color="black"
        )
    )

def setup(title):
    fig, ax = plt.subplots(figsize=(12, 14))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 15)
    ax.axis("off")
    ax.set_title(title, fontsize=20, fontweight="bold", pad=20)
    return fig, ax


# ============================================================
# 1. DESKTOP OPERATING SYSTEM
# ============================================================

fig, ax = setup("Desktop Operating System")

layer(ax, 1, 12.8, 10, 1, "USER SPACE")

box(ax, 1.4, 11.3, 2.7, 0.8, "Applications\nBrowser / IDE / Games")
box(ax, 4.65, 11.3, 2.7, 0.8, "Desktop Environment\nGUI / Shell")
box(ax, 7.9, 11.3, 2.7, 0.8, "User Utilities\nFile Manager / Terminal")

arrow(ax, 2.75, 11.3, 2.75, 10.7)
arrow(ax, 6, 11.3, 6, 10.7)
arrow(ax, 9.25, 11.3, 9.25, 10.7)

layer(ax, 1, 9.5, 10, 0.9, "SYSTEM CALL INTERFACE")

box(ax, 2.1, 7.9, 3.1, 0.8, "System Calls\nPOSIX / Win32")
box(ax, 6.8, 7.9, 3.1, 0.8, "System Libraries\nC Library / APIs")

arrow(ax, 3.1, 9.5, 3.65, 8.7)
arrow(ax, 8.9, 9.5, 8.35, 8.7)

layer(ax, 1, 4.3, 10, 3, "KERNEL SPACE")

box(ax, 1.3, 5.6, 2.3, 0.8, "Process Manager\nCPU Scheduling")
box(ax, 3.95, 5.6, 2.3, 0.8, "Memory Manager\nVirtual Memory")
box(ax, 6.6, 5.6, 2.3, 0.8, "File System\nNTFS / ext4")
box(ax, 3.95, 4.55, 2.3, 0.55, "Device Drivers")
box(ax, 6.6, 4.55, 2.3, 0.55, "I/O Management")

arrow(ax, 3.65, 7.9, 2.45, 6.4)
arrow(ax, 3.65, 7.9, 5.1, 6.4)
arrow(ax, 8.35, 7.9, 7.75, 6.4)
arrow(ax, 5.1, 5.6, 5.1, 5.1)
arrow(ax, 7.75, 5.6, 7.75, 5.1)

layer(ax, 1, 1, 10, 1.5, "HARDWARE LAYER")

box(ax, 1.4, 1.35, 2.2, 0.55, "CPU / Processor")
box(ax, 4.05, 1.35, 2.2, 0.55, "RAM / Memory")
box(ax, 6.7, 1.35, 2.2, 0.55, "Storage")
box(ax, 9.05, 1.35, 1.3, 0.55, "I/O Devices")

arrow(ax, 5.1, 4.55, 2.5, 2.5)
arrow(ax, 5.1, 4.55, 5.15, 2.5)
arrow(ax, 7.75, 4.55, 7.8, 2.5)
arrow(ax, 7.75, 4.55, 9.7, 2.5)

plt.tight_layout()
plt.savefig("desktop_os_flowchart.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()


# ============================================================
# 2. NETWORK OPERATING SYSTEM
# ============================================================

fig, ax = setup("Network Operating System (NOS)")

layer(ax, 1, 12.8, 10, 1, "NETWORK CLIENTS")

box(ax, 1.5, 11.3, 2.5, 0.8, "Client PC 1")
box(ax, 4.75, 11.3, 2.5, 0.8, "Client PC 2")
box(ax, 8, 11.3, 2.5, 0.8, "Admin Console")

arrow(ax, 2.75, 11.3, 2.75, 10.7)
arrow(ax, 6, 11.3, 6, 10.7)
arrow(ax, 9.25, 11.3, 9.25, 10.7)

layer(ax, 1, 9.5, 10, 0.9, "NETWORK OPERATING SYSTEM")

box(ax, 1.2, 7.8, 2.7, 0.9, "Network Services\nDHCP / DNS / NAT", 9)
box(ax, 4.65, 7.8, 2.7, 0.9, "Routing & Switching\nOSPF / BGP / VLAN", 9)
box(ax, 8.1, 7.8, 2.7, 0.9, "Security & Access\nAuthentication / ACL", 9)

arrow(ax, 2.55, 9.5, 2.55, 8.7)
arrow(ax, 6, 9.5, 6, 8.7)
arrow(ax, 9.45, 9.5, 9.45, 8.7)

layer(ax, 1, 6.2, 10, 0.9, "DATA PLANE")

box(
    ax,
    2.2,
    4.65,
    7.6,
    0.9,
    "Packet Processing  •  Forwarding  •  Filtering  •  QoS",
    9
)

arrow(ax, 2.55, 7.8, 4.2, 5.55)
arrow(ax, 6, 7.8, 6, 5.55)
arrow(ax, 9.45, 7.8, 7.8, 5.55)
arrow(ax, 6, 6.2, 6, 5.55)

layer(ax, 1, 3.1, 10, 0.9, "NETWORK HARDWARE")

box(ax, 1.5, 1.4, 2.4, 0.9, "Network\nInterfaces")
box(ax, 4.8, 1.4, 2.4, 0.9, "Switch / Router")
box(ax, 8.1, 1.4, 2.4, 0.9, "Network\nController")

arrow(ax, 6, 4.65, 6, 4)
arrow(ax, 6, 3.1, 2.7, 2.3)
arrow(ax, 6, 3.1, 6, 2.3)
arrow(ax, 6, 3.1, 9.3, 2.3)

plt.tight_layout()
plt.savefig("network_os_flowchart.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()


# ============================================================
# 3. EMBEDDED OPERATING SYSTEM
# ============================================================

fig, ax = setup("Embedded Operating System")

layer(ax, 1, 12.8, 10, 1, "APPLICATION LAYER")

box(ax, 1.2, 11.3, 2.7, 0.9, "Sensor Tasks")
box(ax, 4.65, 11.3, 2.7, 0.9, "Control Tasks")
box(ax, 8.1, 11.3, 2.7, 0.9, "Communication Tasks")

arrow(ax, 2.55, 11.3, 2.55, 10.7)
arrow(ax, 6, 11.3, 6, 10.7)
arrow(ax, 9.45, 11.3, 9.45, 10.7)

layer(ax, 1, 9.5, 10, 0.9, "EMBEDDED OS / RTOS")

box(ax, 1.2, 7.8, 2.7, 0.9, "Task Scheduler\nPriority / Preemption", 9)
box(ax, 4.65, 7.8, 2.7, 0.9, "Inter-Task Communication\nQueues / Mutexes", 8.5)
box(ax, 8.1, 7.8, 2.7, 0.9, "Device Management\nI/O Handling", 9)

arrow(ax, 2.55, 9.5, 2.55, 8.7)
arrow(ax, 6, 9.5, 6, 8.7)
arrow(ax, 9.45, 9.5, 9.45, 8.7)

layer(ax, 1, 6.2, 10, 0.9, "HARDWARE ABSTRACTION LAYER (HAL)")

box(ax, 1.2, 4.65, 2.7, 0.9, "Device Drivers\nUART / SPI / I²C", 9)
box(ax, 4.65, 4.65, 2.7, 0.9, "Board Support Package\n(BSP)", 9)
box(ax, 8.1, 4.65, 2.7, 0.9, "Interrupt Handling\nIRQ / NVIC", 9)

arrow(ax, 2.55, 7.8, 2.55, 5.55)
arrow(ax, 6, 7.8, 6, 5.55)
arrow(ax, 9.45, 7.8, 9.45, 5.55)

arrow(ax, 6, 6.2, 6, 5.55)

layer(ax, 1, 3.1, 10, 0.9, "HARDWARE LAYER")

box(ax, 1.2, 1.4, 2.7, 0.9, "Sensors\nTemperature / Distance")
box(ax, 4.65, 1.4, 2.7, 0.9, "Processor & Memory\nCPU / RAM / Flash")
box(ax, 8.1, 1.4, 2.7, 0.9, "Actuators\nMotors / Relays")

arrow(ax, 2.55, 4.65, 2.55, 2.3)
arrow(ax, 6, 4.65, 6, 2.3)
arrow(ax, 9.45, 4.65, 9.45, 2.3)

plt.tight_layout()
plt.savefig("embedded_os_flowchart.png", dpi=300, bbox_inches="tight")
plt.show()
plt.close()

print("\nAll 3 flowcharts generated successfully:")
print("1. desktop_os_flowchart.png")
print("2. network_os_flowchart.png")
print("3. embedded_os_flowchart.png")