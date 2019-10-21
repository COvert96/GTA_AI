# Citation: Box Of Hats (https://github.com/Box-Of-Hats )

import win32api as wapi
import time

chars = "ABCDEFGHIJKLMNOPQRSTWXYZ 123456789,.'£$/\\"
keyList = ["\b"]
for char in "ABDPSTW ":
    keyList.append(char)

def key_check():
    keys = []
    for key in keyList:
        if wapi.GetAsyncKeyState(ord(key)):
            keys.append(key)
    return keys
 
