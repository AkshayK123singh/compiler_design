from flask import Flask, request, jsonify, send_from_directory
import subprocess
import os
import tempfile
import psutil
import time
import threading

app = Flask(__name__)

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

def monitor_memory(proc, memory_dict):
    try:
        p = psutil.Process(proc.pid)
        while proc.poll() is None:
            mem = p.memory_info().rss  # in bytes
            memory_dict['max'] = max(memory_dict['max'], mem)
            time.sleep(0.05)
    except psutil.NoSuchProcess:
        pass  # the process ended quickly

@app.route('/run_c_code', methods=['POST'])
def run_c_code():
    data = request.get_json()
    code = data.get('code')
    prog_input = data.get('input', '')

    if not code:
        return jsonify({'error': 'No code provided'}), 400

    with tempfile.TemporaryDirectory() as tempdir:
        c_file = os.path.join(tempdir, 'program.c')
        exe_file = os.path.join(tempdir, 'program.exe' if os.name == 'nt' else 'program.out')

        with open(c_file, 'w') as f:
            f.write(code)

        compile_proc = subprocess.run(
            ['gcc', c_file, '-o', exe_file],
            capture_output=True, text=True, timeout=10
        )

        if compile_proc.returncode != 0:
            return jsonify({'error': compile_proc.stderr})

        try:
            memory_usage = {'max': 0}
            start_time = time.time()

            proc = subprocess.Popen(
                [exe_file],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            monitor_thread = threading.Thread(target=monitor_memory, args=(proc, memory_usage))
            monitor_thread.start()

            try:
                stdout, stderr = proc.communicate(input=prog_input, timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                return jsonify({'timeout': True})

            monitor_thread.join()

            end_time = time.time()
            exec_time = round(end_time - start_time, 4)
            peak_mem_kb = memory_usage['max'] // 1024

            if stderr:
                return jsonify({'error': stderr})

            return jsonify({
                'output': stdout,
                'exec_time': exec_time,
                'memory_kb': peak_mem_kb
            })

        except Exception as e:
            return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
