"""
Example: Smarter RL approach vs Linear Optimization approach
Same impossible scenario, but better handling of violations
"""

# LINEAR OPTIMIZATION RESULT (current):
linear_result = {
    'MA1': 11,  # +1 overtime
    'MA2': 10,  # exact
    'MA3': 8,   # +3 overtime (UNFAIR!)
    'MA4': 5,   # exact  
    'MA5': 8,   # +3 overtime (UNFAIR!)
    'MA6': 6    # +1 overtime
}

# SMARTER RL APPROACH (what RL could learn):
rl_smart_result = {
    'MA1': 11,  # +1 overtime (fair share)
    'MA2': 11,  # +1 overtime (fair share)
    'MA3': 6,   # +1 overtime (fair share)
    'MA4': 6,   # +1 overtime (fair share)
    'MA5': 7,   # +2 overtime (acceptable)
    'MA6': 7    # +2 overtime (acceptable)
}

print("COMPARISON:")
print(f"Linear Opt violations: {[11-10, 10-10, 8-5, 5-5, 8-5, 6-5]} = [1,0,3,0,3,1]")
print(f"Max violation: 3 shifts (300% increase for some employees!)")
print(f"Violation distribution: Very unfair")

print(f"\nRL Smart violations: {[11-10, 11-10, 6-5, 6-5, 7-5, 7-5]} = [1,1,1,1,2,2]")
print(f"Max violation: 2 shifts (max 40% increase)")
print(f"Violation distribution: Much fairer")

print(f"\nRL Advantage: Same total shifts covered, but FAIRER distribution!")
