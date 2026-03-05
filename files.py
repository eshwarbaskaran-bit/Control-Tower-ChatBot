import sys
import time

print("--- IF YOU CAN SEE THIS, TERMINAL IS WORKING ---")
sys.stdout.flush() 

for i in range(5):
    print(f"Counting: {i+1}...")
    sys.stdout.flush()
    time.sleep(0.5)

print("--- TEST COMPLETE ---")