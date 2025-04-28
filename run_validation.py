import subprocess
import os
import sys

# Settings
TEST_DIR = 'test_problems/validation/'
N_TRIALS = 5  # Number of times to run each test
ERROR_TOLERANCE = 0.01  # 1% relative error allowed

# Find all test instruction files
test_files = [f for f in os.listdir(TEST_DIR) if f.endswith('.txt') and not f.endswith('_solution.txt')]
test_files.sort()

# Storage for results
correct = {tf[:-4]: 0 for tf in test_files}  # strip .txt

# Run tests
for test_file in test_files:
    test_name = test_file[:-4]
    solution_file = os.path.join(TEST_DIR, test_name + '_solution.txt')

    if not os.path.exists(solution_file):
        print(f"Warning: No solution file for {test_name}, skipping.")
        continue

    # Load expected solution
    with open(solution_file, 'r') as f:
        expected_answer = f.read().strip()

    # Try to interpret as float
    try:
        expected_value = float(expected_answer)
        expected_is_float = True
    except ValueError:
        expected_is_float = False
    
    trial_results = ['-'] * N_TRIALS

    # Run multiple trials
    for trial in range(N_TRIALS):
        trial_results[trial] = '*'  # Mark current trial as in-progress
        sys.stdout.write(f"\rRunning {test_name} trial {trial+1}/{N_TRIALS}; [{''.join(trial_results)}]")
        sys.stdout.flush()
        
        proc = subprocess.run(['python', 'run_agents.py','--end', os.path.join(TEST_DIR, test_file)], capture_output=True, text=True)

        output_lines = proc.stdout.strip().split('\n')
        if not output_lines:
            trial_results[trial] = 'X'
            continue

        final_line = output_lines[-1].strip()

        try:
            output_value = float(final_line)
            output_is_float = True
        except ValueError:
            output_is_float = False

        success = False
        if expected_is_float and output_is_float:
            rel_error = abs(output_value - expected_value) / abs(expected_value)
            if rel_error <= ERROR_TOLERANCE:
                success = True
        else:
            if final_line == expected_answer:
                success = True

        if success:
            correct[test_name] += 1
            trial_results[trial] = '✓'
        else:
            trial_results[trial] = 'X'

    sys.stdout.write(f"\rCompleted {test_name} trial {trial+1}/{N_TRIALS}; [{''.join(trial_results)}]")
    sys.stdout.flush()

    print()  # Newline after each problem

print("\n--- Validation Results ---")
for test_name, num_correct in correct.items():
    print(f"{test_name:<30} {num_correct}/{N_TRIALS} correct")
