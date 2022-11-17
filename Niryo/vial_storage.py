import serial
import time


"""
Address : /dev/ttyACM2
"""

def vial_Entrance(entrance_num):
    """
    control vial storage's Entrance (1,2,3,4,5 : Bottom Site (Empty Vial)) (6,7,8,9,10 : Top Site (Full Vial))
    :param entrance_num: Entrance Number
    :return: None
    """
    arduinoData = serial.Serial('/dev/ttyACM2', 9600)
    time.sleep(2)
    value = str(entrance_num)
    arduinoData.write(value.encode())
    time.sleep(2)
    arduinoData.close()