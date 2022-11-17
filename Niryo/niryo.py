from __future__ import absolute_import, division, print_function, unicode_literals
from niryo_one_tcp_client import *
import Vial_Storage
import time


"""
parameter 1 : move to X direction
parameter 2 : move to Y direction
parameter 3 : move to Z direction
parameter 4 : move to wrist rotate joints
parameter 5 : move to wrist fold joints
parameter 6 : move to gripper totally
"""

orginal = [0.076, -0.005, 0.157, 0.110, 1.225, -0.032]

# Vial Store
Vial_center = [0.112, 0.186, 0.085, -0.05, 0.051, 1.504]

Vial_1_1 = [0.017, 0.238, -0.075, 0, -0.230, 1.579]  # DONE
Vial_1_2 = [0.017, 0.247, -0.030, 0, -0.230, 1.579]  # DONE
Vial_1_3 = [0.031, 0.174, 0.118, 0, 0, 1.534]  # DONE

Vial_2_1 = [0.080, 0.236, -0.075, 0, -0.230, 1.579]  # DONE
Vial_2_2 = [0.080, 0.247, -0.030, 0, -0.230, 1.579]  # DONE
Vial_2_3 = [0.080, 0.174, 0.118, 0, 0, 1.534]  # DONE

Vial_3_1 = [0.142, 0.236, -0.075, 0, -0.230, 1.579]  # DONE
Vial_3_2 = [0.142, 0.247, -0.030, 0, -0.230, 1.579]  # DONE
Vial_3_3 = [0.142, 0.172, 0.118, 0, 0, 1.534]  # DONE

Vial_4_1 = [0.205, 0.233, -0.075, 0, -0.230, 1.579]  # DONE
Vial_4_2 = [0.205, 0.243, -0.030, 0, -0.230, 1.579]  # DONE
Vial_4_3 = [0.205, 0.172, 0.118, 0, 0, 1.534]  # DONE

Vial_5_1 = [0.224, 0.237, -0.071, 0.136, -0.221, 1.212]  # DONE
Vial_5_2 = [0.224, 0.243, -0.026, 0.086, -0.221, 1.212]  # DONE
Vial_5_3 = [0.224, 0.172, 0.118, 0.086, 0, 1.534]  # DONE

Vial_store = [[Vial_1_1, Vial_1_2, Vial_1_3], [Vial_2_1, Vial_2_2, Vial_2_3],
              [Vial_3_1, Vial_3_2, Vial_3_3], [Vial_4_1, Vial_4_2, Vial_4_3],
              [Vial_5_1, Vial_5_2, Vial_5_3]]

# Stiring Store
Stiring_store_center = [0.263, 0.000, 0.157, 0, 0, 0]  # DONE

Stiring_store_1_1 = [0.227, -0.133, 0.035, 0.00, 0.00, 0.00]  # DONE
Stiring_store_1_2 = [0.227, -0.133, 0.009, 0.00, 0.00, 0.00]  # DONE

Stiring_store_2_1 = [0.233, -0.033, 0.035, 0.00, 0.00, 0.00]  # DONE
Stiring_store_2_2 = [0.233, -0.033, 0.009, 0.00, 0.00, 0.00]  # DONE

Stiring_store_3_1 = [0.236, 0.071, 0.035, 0.00, 0.00, 0.00]  # DONE
Stiring_store_3_2 = [0.236, 0.071, 0.009, 0.00, 0.00, 0.00]  # DONE

Stiring_store_4_1 = [0.239, 0.175, 0.035, 0.057, 0.042, 0.00]  # DONE
Stiring_store_4_2 = [0.239, 0.175, 0.009, 0.057, 0.042, 0.00]  # DONE

Stiring_store_5_1 = [0.328, -0.135, 0.035, 0.00, 0.00, 0.00]  # DONE
Stiring_store_5_2 = [0.328, -0.135, 0.009, 0.00, 0.00, 0.00]  # DONE

Stiring_store_6_1 = [0.333, -0.035, 0.035, 0.00, 0.00, 0.00]  # DONE
Stiring_store_6_2 = [0.333, -0.035, 0.009, 0.00, 0.00, 0.00]  # DONE

Stiring_store_7_1 = [0.335, 0.070, 0.035, 0.00, 0.00, 0.00]  # DONE
Stiring_store_7_2 = [0.335, 0.070, 0.009, 0.00, 0.00, 0.00]  # DONE

Stiring_store_8_1 = [0.335, 0.170, 0.035, 0.025, 0.001, 0.002]  # DONE
Stiring_store_8_2 = [0.335, 0.170, 0.009, 0.025, 0.001, 0.002]  # DONE

Stiring_store = [[Stiring_store_1_1, Stiring_store_1_2], [Stiring_store_2_1, Stiring_store_2_2],
                 [Stiring_store_3_1, Stiring_store_3_2], [Stiring_store_4_1, Stiring_store_4_2],
                 [Stiring_store_5_1, Stiring_store_5_2], [Stiring_store_6_1, Stiring_store_6_2],
                 [Stiring_store_7_1, Stiring_store_7_2], [Stiring_store_8_1, Stiring_store_8_2]]

# Vial holder
Vial_holder_center = [0.094, -0.135, 0.112, -0.046, 0.041, -0.892]  # DONE

Vial_holder_1_1 = [-0.131, -0.221, -0.041, 0.00, 0.00, -1.608]  # DONE
Vial_holder_1_2 = [-0.131, -0.222, -0.061, 0.00, 0.00, -1.608]  # DONE

Vial_holder_2_1 = [-0.037, -0.224, -0.041, 0.00, 0.00, -2.609]  # DONE
Vial_holder_2_2 = [-0.037, -0.223, -0.059, 0.00, 0.00, -2.609]  # DONE

Vial_holder_3_1 = [-0.003, -0.257, -0.041, 0.00, 0.00, -2.303]  # DONE
Vial_holder_3_2 = [-0.003, -0.257, -0.062, 0.00, 0.00, -2.303]  # DONE

Vial_holder_4_1 = [0.032, -0.245, -0.046, 0.00, 0.00, -2.786]  # DONE
Vial_holder_4_2 = [0.032, -0.245, -0.070, 0.00, 0.00, -2.786]  # DONE

Vial_holder_5_1 = [0.067, -0.270, -0.046, 0.00, 0.00, -2.436]  # DONE
Vial_holder_5_2 = [0.067, -0.270, -0.064, 0.00, 0.00, -2.436]  # DONE

Vial_holder_6_1 = [0.079, -0.226, -0.041, 0.00, 0.00, -2.584]  # DONE
Vial_holder_6_2 = [0.079, -0.226, -0.066, 0.00, 0.00, -2.584]  # DONE

Vial_holder_7_1 = [0.126, -0.264, -0.041, 0.00, 0.00, -2.397]  # DONE
Vial_holder_7_2 = [0.123, -0.267, -0.065, 0.00, 0.00, -2.3972]  # DONE

Vial_holder_8_1 = [0.126, -0.213, -0.047, 0.00, 0.0, -2.433]  # DONE
Vial_holder_8_2 = [0.126, -0.213, -0.068, 0.00, 0.0, -2.433]  # DONE

VialHolder_store = [[Vial_holder_1_1, Vial_holder_1_2], [Vial_holder_2_1, Vial_holder_2_2],
                    [Vial_holder_3_1, Vial_holder_3_2], [Vial_holder_4_1, Vial_holder_4_2],
                    [Vial_holder_5_1, Vial_holder_5_2], [Vial_holder_6_1, Vial_holder_6_2],
                    [Vial_holder_7_1, Vial_holder_7_2], [Vial_holder_8_1, Vial_holder_8_2]]

# Storageback
StorageBack_center = [0.124, 0.215, 0.272, 0.00, 0.00, 1.103]  # DONE

Vial_holderBack_1_1 = [0.020, 0.344, 0.262, 0.00, 0.00, 1.573]  # DONE
Vial_holderBack_1_2 = [0.020, 0.344, 0.196, 0.00, 0.00, 1.573]  # DONE

Vial_holderBack_2_1 = [0.081, 0.344, 0.262, 0.00, 0.00, 1.573]  # DONE
Vial_holderBack_2_2 = [0.081, 0.344, 0.196, 0.00, 0.00, 1.573]  # DONE

Vial_holderBack_3_1 = [0.141, 0.343, 0.262, 0.00, 0.00, 1.573]  # DONE
Vial_holderBack_3_2 = [0.141, 0.343, 0.196, 0.00, 0.00, 1.573]  # DONE

Vial_holderBack_4_1 = [0.201, 0.339, 0.262, 0.00, 0.00, 1.573]  # DONE
Vial_holderBack_4_2 = [0.201, 0.339, 0.196, 0.00, 0.00, 1.573]  # DONE

Vial_holderBack_5_1 = [0.217, 0.347, 0.262, 0.00, 0.00, 1.173]  # DONE
Vial_holderBack_5_2 = [0.217, 0.347, 0.196, 0.00, 0.00, 1.173]  # DONE


Vial_holderBack_store = [[Vial_holderBack_1_1, Vial_holderBack_1_2], [Vial_holderBack_2_1, Vial_holderBack_2_2],
                         [Vial_holderBack_3_1, Vial_holderBack_3_2], [Vial_holderBack_4_1, Vial_holderBack_4_2],
                         [Vial_holderBack_5_1, Vial_holderBack_5_2]]


class Niryo(object):
    def __init__(self):
        self.niryo_one = NiryoOneClient()
        self.niryo_one.connect("169.254.200.200")
        self.niryo_one.change_tool(RobotTool.GRIPPER_2)
        self.niryo_one.set_arm_max_velocity(60)

    def set_calibrate(self):
        status, data = self.niryo_one.calibrate(CalibrateMode.MANUAL)
        if status is False:
            print("Error for calibrate mode: " + data)

    def move_vialStorage_to_stiring(self, vial_loc=1, stiring_loc=1):
        Vial_Storage.vial_Entrance(vial_loc)
        time.sleep(0.5)
        self.niryo_one.set_arm_max_velocity(60)
        time.sleep(0.5)
        self.niryo_one.move_pose(float(orginal[0]), float(orginal[1]), float(orginal[2]),
                                 float(orginal[3]), float(orginal[4]), float(orginal[5]))
        time.sleep(0.5)
        self.niryo_one.open_gripper(RobotTool.GRIPPER_2, 300)
        time.sleep(0.5)
        self.niryo_one.move_pose(float(Vial_center[0]), float(Vial_center[1]), float(Vial_center[2]),
                                 float(Vial_center[3]), float(Vial_center[4]), float(Vial_center[5]))
        time.sleep(1)
        self.niryo_one.move_pose(float(Vial_store[int(vial_loc) - 1][0][0]),
                                 float(Vial_store[int(vial_loc) - 1][0][1]),
                                 float(Vial_store[int(vial_loc) - 1][0][2]),
                                 float(Vial_store[int(vial_loc) - 1][0][3]),
                                 float(Vial_store[int(vial_loc) - 1][0][4]),
                                 float(Vial_store[int(vial_loc) - 1][0][5]))
        time.sleep(0.5)
        self.niryo_one.close_gripper(RobotTool.GRIPPER_2, 400)
        time.sleep(0.5)
        self.niryo_one.close_gripper(RobotTool.GRIPPER_2, 400)
        time.sleep(0.5)
        self.niryo_one.move_pose(float(Vial_store[int(vial_loc) - 1][1][0]),
                                 float(Vial_store[int(vial_loc) - 1][1][1]),
                                 float(Vial_store[int(vial_loc) - 1][1][2]),
                                 float(Vial_store[int(vial_loc) - 1][1][3]),
                                 float(Vial_store[int(vial_loc) - 1][1][4]),
                                 float(Vial_store[int(vial_loc) - 1][1][5]))
        time.sleep(0.5)
        self.niryo_one.move_pose(float(Vial_store[int(vial_loc) - 1][2][0]),
                                 float(Vial_store[int(vial_loc) - 1][2][1]),
                                 float(Vial_store[int(vial_loc) - 1][2][2]),
                                 float(Vial_store[int(vial_loc) - 1][2][3]),
                                 float(Vial_store[int(vial_loc) - 1][2][4]),
                                 float(Vial_store[int(vial_loc) - 1][2][5]))
        time.sleep(1)
        self.niryo_one.move_pose(float(Stiring_store_center[0]), float(Stiring_store_center[1]),
                                 float(Stiring_store_center[2]), float(Stiring_store_center[3]),
                                 float(Stiring_store_center[4]), float(Stiring_store_center[5]))
        time.sleep(1)
        self.niryo_one.move_pose(float(Stiring_store[int(stiring_loc) - 1][0][0]),
                                 float(Stiring_store[int(stiring_loc) - 1][0][1]),
                                 float(Stiring_store[int(stiring_loc) - 1][0][2]),
                                 float(Stiring_store[int(stiring_loc) - 1][0][3]),
                                 float(Stiring_store[int(stiring_loc) - 1][0][4]),
                                 float(Stiring_store[int(stiring_loc) - 1][0][5]))
        time.sleep(1)
        self.niryo_one.move_pose(float(Stiring_store[int(stiring_loc) - 1][1][0]),
                                 float(Stiring_store[int(stiring_loc) - 1][1][1]),
                                 float(Stiring_store[int(stiring_loc) - 1][1][2]),
                                 float(Stiring_store[int(stiring_loc) - 1][1][3]),
                                 float(Stiring_store[int(stiring_loc) - 1][1][4]),
                                 float(Stiring_store[int(stiring_loc) - 1][1][5]))
        time.sleep(0.5)
        self.niryo_one.open_gripper(RobotTool.GRIPPER_2, 300)
        time.sleep(0.5)
        self.niryo_one.move_pose(float(orginal[0]), float(orginal[1]), float(orginal[2]),
                                 float(orginal[3]), float(orginal[4]), float(orginal[5]))

    def move_stiring_to_vialHolder(self, stiring_loc=1, vialHolder_loc=1):
        self.niryo_one.set_arm_max_velocity(60)
        time.sleep(0.5)
        self.niryo_one.open_gripper(RobotTool.GRIPPER_2, 300)
        time.sleep(0.5)
        self.niryo_one.move_pose(float(orginal[0]), float(orginal[1]), float(orginal[2]), float(orginal[3]),
                                 float(orginal[4]), float(orginal[5]))
        time.sleep(1)
        self.niryo_one.move_pose(float(Stiring_store[int(stiring_loc) - 1][1][0]),
                                 float(Stiring_store[int(stiring_loc) - 1][1][1]),
                                 float(Stiring_store[int(stiring_loc) - 1][1][2]),
                                 float(Stiring_store[int(stiring_loc) - 1][1][3]),
                                 float(Stiring_store[int(stiring_loc) - 1][1][4]),
                                 float(Stiring_store[int(stiring_loc) - 1][1][5]))
        time.sleep(0.5)
        self.niryo_one.close_gripper(RobotTool.GRIPPER_2, 300)
        time.sleep(0.5)
        self.niryo_one.move_pose(float(Stiring_store[int(stiring_loc) - 1][0][0]),
                                 float(Stiring_store[int(stiring_loc) - 1][0][1]),
                                 float(Stiring_store[int(stiring_loc) - 1][0][2]),
                                 float(Stiring_store[int(stiring_loc) - 1][0][3]),
                                 float(Stiring_store[int(stiring_loc) - 1][0][4]),
                                 float(Stiring_store[int(stiring_loc) - 1][0][5]))
        time.sleep(1)
        self.niryo_one.move_pose(float(Vial_holder_center[0]), float(Vial_holder_center[1]),
                                 float(Vial_holder_center[2]), float(Vial_holder_center[3]),
                                 float(Vial_holder_center[4]), float(Vial_holder_center[5]))
        time.sleep(1)
        self.niryo_one.move_pose(float(VialHolder_store[int(vialHolder_loc) - 1][0][0]),
                                 float(VialHolder_store[int(vialHolder_loc) - 1][0][1]),
                                 float(VialHolder_store[int(vialHolder_loc) - 1][0][2]),
                                 float(VialHolder_store[int(vialHolder_loc) - 1][0][3]),
                                 float(VialHolder_store[int(vialHolder_loc) - 1][0][4]),
                                 float(VialHolder_store[int(vialHolder_loc) - 1][0][5]))
        time.sleep(0.5)
        self.niryo_one.move_pose(float(VialHolder_store[int(vialHolder_loc) - 1][1][0]),
                                 float(VialHolder_store[int(vialHolder_loc) - 1][1][1]),
                                 float(VialHolder_store[int(vialHolder_loc) - 1][1][2]),
                                 float(VialHolder_store[int(vialHolder_loc) - 1][1][3]),
                                 float(VialHolder_store[int(vialHolder_loc) - 1][1][4]),
                                 float(VialHolder_store[int(vialHolder_loc) - 1][1][5]))
        time.sleep(0.5)
        self.niryo_one.open_gripper(RobotTool.GRIPPER_2, 300)
        time.sleep(0.5)
        self.niryo_one.move_pose(float(Vial_holder_center[0]), float(Vial_holder_center[1]),
                                 float(Vial_holder_center[2]), float(Vial_holder_center[3]),
                                 float(Vial_holder_center[4]), float(Vial_holder_center[5]))
        time.sleep(0.5)
        self.niryo_one.move_pose(float(orginal[0]), float(orginal[1]), float(orginal[2]), float(orginal[3]),
                                 float(orginal[4]), float(orginal[5]))

    def move_vialHolder_to_storageBack(self, vialHolder_loc=1, vialBack_loc=1):
        self.niryo_one.set_arm_max_velocity(60)
        time.sleep(0.5)
        self.niryo_one.open_gripper(RobotTool.GRIPPER_2, 300)
        time.sleep(0.5)
        self.niryo_one.move_pose(float(orginal[0]), float(orginal[1]), float(orginal[2]), float(orginal[3]),
                                 float(orginal[4]), float(orginal[5]))
        time.sleep(0.5)
        self.niryo_one.move_pose(float(Vial_holder_center[0]), float(Vial_holder_center[1]),
                                 float(Vial_holder_center[2]), float(Vial_holder_center[3]),
                                 float(Vial_holder_center[4]), float(Vial_holder_center[5]))
        time.sleep(0.5)
        self.niryo_one.move_pose(float(VialHolder_store[int(vialHolder_loc) - 1][1][0]),
                                 float(VialHolder_store[int(vialHolder_loc) - 1][1][1]),
                                 float(VialHolder_store[int(vialHolder_loc) - 1][1][2]),
                                 float(VialHolder_store[int(vialHolder_loc) - 1][1][3]),
                                 float(VialHolder_store[int(vialHolder_loc) - 1][1][4]),
                                 float(VialHolder_store[int(vialHolder_loc) - 1][1][5]))
        time.sleep(0.5)
        self.niryo_one.close_gripper(RobotTool.GRIPPER_2, 300)
        time.sleep(0.5)
        self.niryo_one.close_gripper(RobotTool.GRIPPER_2, 300)
        time.sleep(0.5)
        self.niryo_one.move_pose(float(VialHolder_store[int(vialHolder_loc) - 1][0][0]),
                                 float(VialHolder_store[int(vialHolder_loc) - 1][0][1]),
                                 float(VialHolder_store[int(vialHolder_loc) - 1][0][2]),
                                 float(VialHolder_store[int(vialHolder_loc) - 1][0][3]),
                                 float(VialHolder_store[int(vialHolder_loc) - 1][0][4]),
                                 float(VialHolder_store[int(vialHolder_loc) - 1][0][5]))
        time.sleep(0.5)
        self.niryo_one.move_pose(float(Vial_holder_center[0]), float(Vial_holder_center[1]),
                                 float(Vial_holder_center[2]), float(Vial_holder_center[3]),
                                 float(Vial_holder_center[4]), float(Vial_holder_center[5]))
        time.sleep(1)
        self.niryo_one.move_pose(float(StorageBack_center[0]), float(StorageBack_center[1]),
                                 float(StorageBack_center[2]), float(StorageBack_center[3]),
                                 float(StorageBack_center[4]), float(StorageBack_center[5]))
        time.sleep(1)
        self.niryo_one.move_pose(float(Vial_holderBack_store[int(vialBack_loc) - 1][0][0]),
                                 float(Vial_holderBack_store[int(vialBack_loc) - 1][0][1]),
                                 float(Vial_holderBack_store[int(vialBack_loc) - 1][0][2]),
                                 float(Vial_holderBack_store[int(vialBack_loc) - 1][0][3]),
                                 float(Vial_holderBack_store[int(vialBack_loc) - 1][0][4]),
                                 float(Vial_holderBack_store[int(vialBack_loc) - 1][0][5]))
        time.sleep(1)
        self.niryo_one.move_pose(float(Vial_holderBack_store[int(vialBack_loc) - 1][1][0]),
                                 float(Vial_holderBack_store[int(vialBack_loc) - 1][1][1]),
                                 float(Vial_holderBack_store[int(vialBack_loc) - 1][1][2]),
                                 float(Vial_holderBack_store[int(vialBack_loc) - 1][1][3]),
                                 float(Vial_holderBack_store[int(vialBack_loc) - 1][1][4]),
                                 float(Vial_holderBack_store[int(vialBack_loc) - 1][1][5]))
        time.sleep(0.5)
        self.niryo_one.open_gripper(RobotTool.GRIPPER_2, 300)
        time.sleep(0.5)
        self.niryo_one.move_pose(float(orginal[0]), float(orginal[1]), float(orginal[2]), float(orginal[3]),
                                 float(orginal[4]), float(orginal[5]))
        Vial_Storage.vial_Entrance(int(vialBack_loc)+5)