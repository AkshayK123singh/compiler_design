from flask import Flask, request, jsonify, send_from_directory
import subprocess
import os
import tempfile
import psutil
import time
import threading
import re
import json
from datetime import datetime

# Try to import resource module (available on Unix-like systems)
try:
    import resource
except ImportError:
    # On Windows, create a dummy resource module with the required attributes
    class DummyResource:
        def getrusage(self, who):
            return type('obj', (object,), {'ru_maxrss': 0})
        RUSAGE_CHILDREN = 0
    resource = DummyResource()

app = Flask(__name__)

# Define the absolute path to your HTML files
# Ensure this path is correct relative to your WSL environment where the Flask app runs
HOME_PAGE_DIR = '/mnt/c/Users/yashm/OneDrive/Desktop/front'
HOME_PAGE_FILENAME = 'front.html' # This is your main landing page (homepage)

# --- Debugging: Check if the main home page path exists at server startup ---
HOME_PAGE_FULL_PATH = os.path.join(HOME_PAGE_DIR, HOME_PAGE_FILENAME)
if not os.path.exists(HOME_PAGE_FULL_PATH):
    print(f"ERROR: Main home page file not found at: {HOME_PAGE_FULL_PATH}")
    print("Please ensure the path is correct and the file exists in the WSL environment.")
else:
    print(f"INFO: Main home page file found at: {HOME_PAGE_FULL_PATH}")

@app.route('/')
def serve_front_page():
    # Serves front.html when the root URL is accessed
    try:
        return send_from_directory(HOME_PAGE_DIR, HOME_PAGE_FILENAME)
    except Exception as e:
        print(f"Error serving {HOME_PAGE_FILENAME} from {HOME_PAGE_DIR}: {e}")
        return jsonify({'error': f'Failed to serve home page: {str(e)}'}), 500

@app.route('/front.html')
def serve_front_page_direct():
    """Serves the front.html page when directly accessed."""
    try:
        return send_from_directory(HOME_PAGE_DIR, 'front.html')
    except Exception as e:
        print(f"Error serving front.html from {HOME_PAGE_DIR}: {e}")
        return jsonify({'error': f'Failed to serve home page: {str(e)}'}), 500

# --- Routes for other HTML pages to enable navigation ---

@app.route('/about.html')
def serve_about_page():
    """Serves the about.html page."""
    try:
        return send_from_directory(HOME_PAGE_DIR, 'about.html')
    except Exception as e:
        print(f"Error serving about.html from {HOME_PAGE_DIR}: {e}")
        return jsonify({'error': f'Failed to serve about page: {str(e)}'}), 500

@app.route('/index.html')
def serve_index_page():
    """Serves the index.html page (compiler interface)."""
    try:
        return send_from_directory(HOME_PAGE_DIR, 'index.html')
    except Exception as e:
        print(f"Error serving index.html from {HOME_PAGE_DIR}: {e}")
        return jsonify({'error': f'Failed to serve compiler page: {str(e)}'}), 500

@app.route('/keyfeature.html')
def serve_keyfeature_page():
    """Serves the keyfeature.html page."""
    try:
        return send_from_directory(HOME_PAGE_DIR, 'keyfeature.html')
    except Exception as e:
        print(f"Error serving keyfeature.html from {HOME_PAGE_DIR}: {e}")
        return jsonify({'error': f'Failed to serve features page: {str(e)}'}), 500

PARSER_PATH = './parser'

if not os.path.exists(PARSER_PATH):
    print(f"Warning: JIT Parser executable not found at {PARSER_PATH}. Please ensure it's compiled and placed correctly.")
    print("You might need to run 'chmod +x ./parser' if it's a new executable.")

# --- Updated Output Filtering ---
def filter_program_output(output: str) -> str:
    output = output.strip()
    lines = output.splitlines()

    # Remove lines that are just symbols or empty
    lines = [line for line in lines if line.strip() not in ['}', '{', ']', '[', ')', '('] and line.strip() != '']

    patterns_to_remove = [
        r'^;.*$',                       # LLVM comments
        r'^source_filename.*$',
        r'^target datalayout.*$',
        r'^target triple.*$',
        r'^declare i\d+ @.*$',
        r'^define i\d+ @.*$',
        r'^entry:.*$',
        r'^@\w+ = private unnamed_addr.*$',    # Global IR
        r'^attributes #\d+ =.*$',
        r'!llvm\..*$',                  # LLVM metadata
        r'^ret i\d+.*$',
        r'^Execution Result:.*$',
        r'^%.* = .*$',                  # LLVM IR instruction
        r'^br label.*$',                # branch instruction
        r'^br i1 .*$',                  # conditional branch
        r'^store .*$',                  # store instruction
        r'^load .*$',                   # load instruction
        r'^\s*$',                       # empty line
        r'^\w+: *; preds = .*$',        # LLVM basic block labels like "forcond: ; preds = ..."
    ]

    filtered_lines = []
    for line in lines:
        stripped_line = line.strip()
        if any(re.match(pattern, stripped_line) for pattern in patterns_to_remove):
            continue
        filtered_lines.append(stripped_line)

    return "\n".join(filtered_lines) if filtered_lines else "No clear program output detected."

# --- Monitor memory ---
def monitor_memory(proc, memory_dict):
    """
    Monitors the peak memory usage (RSS) of a given process aggressively.
    """
    try:
        p = psutil.Process(proc.pid)
        while proc.poll() is None: # While the process is still running
            try:
                mem = p.memory_info().rss
                memory_dict['max'] = max(memory_dict['max'], mem)
            except psutil.NoSuchProcess:
                # Process might have just exited between poll() and memory_info()
                break
            # No sleep here for aggressive monitoring
    except psutil.NoSuchProcess:
        pass # Process might have exited before psutil.Process(pid) could be created

# --- Run All Compilers Endpoint ---
@app.route('/run_all_compilers', methods=['POST'])
def run_all_compilers():
    data = request.get_json()
    code = data.get('code')
    prog_input = data.get('input', '')

    if not code:
        return jsonify({'error': 'No code provided'}), 400

    results = {
        'jit': {
            'output': '',
            'exec_time': 'N/A',
            'memory_kb': 'N/A',
            'error': None,
            'compile_time': 'N/A',
            'cpu_user_time': 'N/A',
            'cpu_system_time': 'N/A',
            'io_read_bytes': 'N/A',
            'io_write_bytes': 'N/A',
            'num_threads': 'N/A'
        },
        'gcc': {
            'output': '',
            'exec_time': 'N/A',
            'memory_kb': 'N/A',
            'error': None,
            'compile_time': 'N/A',
            'cpu_user_time': 'N/A',
            'cpu_system_time': 'N/A',
            'io_read_bytes': 'N/A',
            'io_write_bytes': 'N/A',
            'num_threads': 'N/A'
        },
        'raw_execution_data_file': '' # To store the content of the data file
    }

    # --- JIT ---
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.c') as temp_file:
        temp_file.write(code)
        temp_file_path = temp_file.name

    try:
        # For JIT, compilation time is inherently part of execution time.
        # We will set jit_compile_time to jit_exec_time after it's calculated.

        jit_start_time = time.time()
        jit_proc_handle = subprocess.Popen( # Use Popen to allow psutil monitoring
            [PARSER_PATH, temp_file_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        jit_process = None
        try:
            # Attempt to get psutil.Process object
            jit_process = psutil.Process(jit_proc_handle.pid)
            jit_memory_usage = {'max': 0}
            jit_monitor_thread = threading.Thread(target=monitor_memory, args=(jit_proc_handle, jit_memory_usage))
            jit_monitor_thread.start()

            stdout, stderr = "", ""
            try:
                stdout, stderr = jit_proc_handle.communicate(input=prog_input, timeout=10)
            except subprocess.TimeoutExpired:
                jit_proc_handle.kill()
                stdout, stderr = jit_proc_handle.communicate()
                results['jit']['error'] = "Error: JIT Compiler execution timed out (max 10 seconds)."
                results['jit']['output'] = "JIT Execution timed out."
            finally:
                jit_monitor_thread.join()

            jit_end_time = time.time()

            results['jit']['exec_time'] = round(jit_end_time - jit_start_time, 4)
            results['jit']['output'] = filter_program_output(stdout)

            # Set JIT compile time equal to its execution time
            results['jit']['compile_time'] = results['jit']['exec_time']

            if stderr:
                results['jit']['error'] = f"JIT Stderr:\n{stderr}"

            # Capture additional metrics for JIT
            try:
                results['jit']['memory_kb'] = jit_memory_usage['max'] // 1024
            except psutil.NoSuchProcess: # Specific handling for process termination
                results['jit']['memory_kb'] = 0
            except Exception as e: # Catch any other exception, set to N/A, and log
                results['jit']['memory_kb'] = 'N/A'
                results['jit']['error'] = (results['jit']['error'] or "") + f"\nError collecting JIT memory: {str(e)}"

            try:
                # Use resource module for CPU times if available and more precise
                rusage = resource.getrusage(resource.RUSAGE_CHILDREN) # Get CPU times for child processes
                results['jit']['cpu_user_time'] = round(rusage.ru_utime, 4)
                results['jit']['cpu_system_time'] = round(rusage.ru_stime, 4)
            except NameError: # resource module not available (e.g., on Windows)
                try:
                    cpu_times = jit_process.cpu_times()
                    results['jit']['cpu_user_time'] = round(cpu_times.user, 4)
                    results['jit']['cpu_system_time'] = round(cpu_times.system, 4)
                except psutil.NoSuchProcess: # Specific handling for process termination
                    results['jit']['cpu_user_time'] = 0
                    results['jit']['cpu_system_time'] = 0
                except Exception as e: # Catch any other exception, set to N/A, and log
                    results['jit']['cpu_user_time'] = 'N/A'
                    results['jit']['cpu_system_time'] = 'N/A'
                    results['jit']['error'] = (results['jit']['error'] or "") + f"\nError collecting JIT CPU times: {str(e)}"
            except Exception as e: # Catch any other exception, set to N/A, and log
                results['jit']['cpu_user_time'] = 'N/A'
                results['jit']['cpu_system_time'] = 'N/A'
                results['jit']['error'] = (results['jit']['error'] or "") + f"\nError collecting JIT CPU times: {str(e)}"

            try:
                io_counters = jit_process.io_counters()
                results['jit']['io_read_bytes'] = io_counters.read_bytes
                results['jit']['io_write_bytes'] = io_counters.write_bytes
            except psutil.NoSuchProcess: # Specific handling for process termination
                results['jit']['io_read_bytes'] = 0
                results['jit']['io_write_bytes'] = 0
            except Exception as e: # Catch any other exception, set to N/A, and log
                results['jit']['io_read_bytes'] = 'N/A'
                results['jit']['io_write_bytes'] = 'N/A'
                results['jit']['error'] = (results['jit']['error'] or "") + f"\nError collecting JIT I/O counters: {str(e)}"

            try:
                results['jit']['num_threads'] = jit_process.num_threads()
            except psutil.NoSuchProcess: # Specific handling for process termination
                results['jit']['num_threads'] = 0
            except Exception as e: # Catch any other exception, set to N/A, and log
                results['jit']['num_threads'] = 'N/A'
                results['jit']['error'] = (results['jit']['error'] or "") + f"\nError collecting JIT threads: {str(e)}"

        except psutil.NoSuchProcess:
            # This outer block catches if psutil.Process(pid) itself fails.
            # Metrics are already defaulted to 0 by the inner blocks if the process
            # terminates during collection, so no need to explicitly set to N/A or 0 here again.
            pass
        except Exception as e:
            results['jit']['error'] = (results['jit']['error'] or "") + f"\nAn unexpected error occurred during JIT process monitoring: {str(e)}"
            print(f"DEBUG: JIT Process Monitoring Error: {str(e)}")

    except FileNotFoundError:
        results['jit']['error'] = f"Error: JIT Compiler executable '{PARSER_PATH}' not found."
        results['jit']['output'] = "JIT Compiler not found."
        print(f"DEBUG: JIT Compiler Not Found Error: {results['jit']['error']}")
    except subprocess.TimeoutExpired:
        results['jit']['error'] = "Error: JIT Compiler execution timed out (max 10 seconds)."
        results['jit']['output'] = "JIT Execution timed out."
        print(f"DEBUG: JIT Execution Timeout Error: {results['jit']['error']}")
    except Exception as e:
        results['jit']['error'] = f"An unexpected error occurred during JIT compilation: {str(e)}"
        results['jit']['output'] = "An unexpected error occurred."
        print(f"DEBUG: JIT Compilation (Unexpected) Error: {str(e)}")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

    # --- GCC ---
    # NEW: Check if gcc is available before attempting compilation
    try:
        subprocess.run(['gcc', '--version'], check=True, capture_output=True, text=True, timeout=5)
        print("DEBUG: gcc executable found in PATH.")
    except FileNotFoundError:
        results['gcc']['error'] = "Error: 'gcc' command not found. Please ensure GCC is installed and in your system's PATH in WSL."
        results['gcc']['output'] = "GCC Compiler not found."
        print(f"DEBUG: GCC Not Found Error: {results['gcc']['error']}")
        # Skip further GCC processing if gcc is not found
        return jsonify(results)
    except subprocess.TimeoutExpired:
        results['gcc']['error'] = "Error: 'gcc --version' command timed out. GCC might be installed but unresponsive."
        results['gcc']['output'] = "GCC Check timed out."
        print(f"DEBUG: GCC Version Check Timeout Error: {results['gcc']['error']}")
        return jsonify(results)
    except subprocess.CalledProcessError as e:
        results['gcc']['error'] = f"Error checking GCC version: {e.stderr}. GCC might be installed but facing issues."
        results['gcc']['output'] = "GCC Check failed."
        print(f"DEBUG: GCC Version Check Failed Error: {e.stderr}")
        return jsonify(results)
    except Exception as e:
        results['gcc']['error'] = f"An unexpected error occurred while checking GCC availability: {str(e)}"
        results['gcc']['output'] = "GCC Check failed."
        print(f"DEBUG: GCC Availability Check (Unexpected) Error: {str(e)}")
        return jsonify(results)


    with tempfile.TemporaryDirectory() as tempdir:
        c_file = os.path.join(tempdir, 'program.c')
        exe_file = os.path.join(tempdir, 'program.exe' if os.name == 'nt' else 'program.out')

        with open(c_file, 'w') as f:
            f.write(code)

        try:
            gcc_compile_start = time.time()
            compile_proc = subprocess.run(
                ['gcc', c_file, '-o', exe_file],
                capture_output=True, text=True, timeout=10
            )
            gcc_compile_end = time.time()
            results['gcc']['compile_time'] = round(gcc_compile_end - gcc_compile_start, 4)

            if compile_proc.returncode != 0:
                results['gcc']['error'] = f"GCC Compilation Error:\n{compile_proc.stderr}"
                results['gcc']['output'] = "GCC Compilation failed."
                print(f"DEBUG: GCC Compile Process Failed (Return Code {compile_proc.returncode}): {compile_proc.stderr}")
            else:
                try:
                    memory_usage = {'max': 0}
                    exec_start_time = time.time()

                    proc_handle = subprocess.Popen(
                        [exe_file],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )

                    gcc_process = None
                    try:
                        gcc_process = psutil.Process(proc_handle.pid)
                        monitor_thread = threading.Thread(target=monitor_memory, args=(proc_handle, memory_usage))
                        monitor_thread.start()

                        stdout, stderr = "", ""
                        try:
                            stdout, stderr = proc_handle.communicate(input=prog_input, timeout=5)
                        except subprocess.TimeoutExpired:
                            proc_handle.kill()
                            stdout, stderr = proc_handle.communicate()
                            results['gcc']['error'] = "GCC Execution timed out (max 5 seconds)."
                            results['gcc']['output'] = "Execution timed out."
                            print(f"DEBUG: GCC Execution Timeout Error: {results['gcc']['error']}")
                        finally:
                            monitor_thread.join()

                        exec_end_time = time.time()
                        results['gcc']['exec_time'] = round(exec_end_time - exec_start_time, 4)
                        results['gcc']['output'] = filter_program_output(stdout)

                        if stderr:
                            results['gcc']['error'] = f"GCC Runtime Error:\n{stderr}"
                            print(f"DEBUG: GCC Runtime Stderr: {stderr}")

                        # Capture additional metrics for GCC
                        try:
                            results['gcc']['memory_kb'] = memory_usage['max'] // 1024
                        except psutil.NoSuchProcess:
                            results['gcc']['memory_kb'] = 0
                        except Exception as e:
                            results['gcc']['memory_kb'] = 'N/A'
                            results['gcc']['error'] = (results['gcc']['error'] or "") + f"\nError collecting GCC memory: {str(e)}"
                            print(f"DEBUG: GCC Memory Collection Error: {str(e)}")

                        try:
                            rusage = resource.getrusage(resource.RUSAGE_CHILDREN)
                            results['gcc']['cpu_user_time'] = round(rusage.ru_utime, 4)
                            results['gcc']['cpu_system_time'] = round(rusage.ru_stime, 4)
                        except NameError:
                            try:
                                cpu_times = gcc_process.cpu_times()
                                results['gcc']['cpu_user_time'] = round(cpu_times.user, 4)
                                results['gcc']['cpu_system_time'] = round(cpu_times.system, 4)
                            except psutil.NoSuchProcess:
                                results['gcc']['cpu_user_time'] = 0
                                results['gcc']['cpu_system_time'] = 0
                            except Exception as e:
                                results['gcc']['cpu_user_time'] = 'N/A'
                                results['gcc']['cpu_system_time'] = 'N/A'
                                results['gcc']['error'] = (results['gcc']['error'] or "") + f"\nError collecting GCC CPU times: {str(e)}"
                                print(f"DEBUG: GCC CPU Times Collection Error: {str(e)}")
                        except Exception as e:
                            results['gcc']['cpu_user_time'] = 'N/A'
                            results['gcc']['cpu_system_time'] = 'N/A'
                            results['gcc']['error'] = (results['gcc']['error'] or "") + f"\nError collecting GCC CPU times: {str(e)}"
                            print(f"DEBUG: GCC CPU Times Collection Error: {str(e)}")

                        try:
                            io_counters = gcc_process.io_counters()
                            results['gcc']['io_read_bytes'] = io_counters.read_bytes
                            results['gcc']['io_write_bytes'] = io_counters.write_bytes
                        except psutil.NoSuchProcess:
                            results['gcc']['io_read_bytes'] = 0
                            results['gcc']['io_write_bytes'] = 0
                        except Exception as e:
                            results['gcc']['io_read_bytes'] = 'N/A'
                            results['gcc']['io_write_bytes'] = 'N/A'
                            results['gcc']['error'] = (results['gcc']['error'] or "") + f"\nError collecting GCC I/O counters: {str(e)}"
                            print(f"DEBUG: GCC I/O Counters Collection Error: {str(e)}")

                        try:
                            results['gcc']['num_threads'] = gcc_process.num_threads()
                        except psutil.NoSuchProcess:
                            results['gcc']['num_threads'] = 0
                        except Exception as e:
                            results['gcc']['num_threads'] = 'N/A'
                            results['gcc']['error'] = (results['gcc']['error'] or "") + f"\nError collecting GCC threads: {str(e)}"
                            print(f"DEBUG: GCC Thread Count Collection Error: {str(e)}")

                    except psutil.NoSuchProcess:
                        pass
                    except Exception as e:
                        results['gcc']['error'] = (results['gcc']['error'] or "") + f"\nAn unexpected error occurred during GCC process monitoring: {str(e)}"
                        print(f"DEBUG: GCC Process Monitoring (Unexpected) Error: {str(e)}")

                except FileNotFoundError:
                    results['gcc']['error'] = f"Error: Compiled GCC executable '{exe_file}' not found. This might indicate a compilation issue."
                    results['gcc']['output'] = "GCC Execution failed."
                    print(f"DEBUG: Compiled GCC Executable Not Found Error: {results['gcc']['error']}")
                except Exception as e:
                    results['gcc']['error'] = f"An unexpected error occurred during GCC execution: {str(e)}"
                    results['gcc']['output'] = "An unexpected error occurred."
                    print(f"DEBUG: GCC Execution (Unexpected) Error: {str(e)}")

        except subprocess.TimeoutExpired:
            results['gcc']['error'] = "GCC Compilation timed out (max 10 seconds)."
            results['gcc']['output'] = "GCC Compilation timed out."
            print(f"DEBUG: GCC Compilation Timeout Error: {results['gcc']['error']}")
        except Exception as e:
            results['gcc']['error'] = f"An unexpected error occurred during GCC compilation: {str(e)}"
            results['gcc']['output'] = "An unexpected error occurred."
            print(f"DEBUG: GCC Compilation (Unexpected) Error: {str(e)}")

    # --- Store all results in a temporary text file ---
    results_for_file = results.copy()
    results_for_file.pop('raw_execution_data_file', None)

    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt', encoding='utf-8') as data_file:
        json.dump(results_for_file, data_file, indent=2)
        data_file_path = data_file.name

    with open(data_file_path, 'r', encoding='utf-8') as data_file_read:
        results['raw_execution_data_file'] = data_file_read.read()

    if os.path.exists(data_file_path):
        os.remove(data_file_path)

    return jsonify(results)

@app.route('/save_current_run', methods=['POST'])
def save_current_run():
    """
    Saves the current run's code, JIT details, and GCC details to a JSON file.
    """
    data = request.get_json()
    code_content = data.get('code', 'No code provided.')
    jit_details = data.get('jit_details', {})
    gcc_details = data.get('gcc_details', {})

    if not data:
        return jsonify({'status': 'error', 'message': 'No data provided to save.'}), 400

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Save the file in the current working directory of the Flask app
    file_name = f"compiler_run_{timestamp}.json"
    
    # Structure the data as requested: code, then JIT, then GCC
    save_data = {
        'code': code_content,
        'jit_details': jit_details,
        'gcc_details': gcc_details
    }

    try:
        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=4)
        return jsonify({'status': 'success', 'message': f'Run data saved to {file_name}'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Failed to save run data: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True)