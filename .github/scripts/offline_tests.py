"""Execute selected Python tests offline, preserving real failures and counts."""
from __future__ import annotations
import os, pathlib, subprocess, sys, tempfile
import xml.etree.ElementTree as ET
GUARD = """import ipaddress, socket
_original = socket.socket.connect
def checked(self, address):
    host = address[0] if isinstance(address, tuple) else None
    if host is not None and host not in ('localhost','127.0.0.1','::1'):
        try: local = ipaddress.ip_address(host).is_loopback
        except ValueError: local = False
        if not local: raise RuntimeError('OFFLINE_TEST: external connection blocked')
    return _original(self,address)
socket.socket.connect = checked
"""
def main():
    repo = pathlib.Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix='offline-validation-') as temp:
        guard = pathlib.Path(temp); (guard/'sitecustomize.py').write_text(GUARD,encoding='utf-8')
        env = {k:v for k,v in os.environ.items() if not any(w in k.upper() for w in ['API_KEY','TOKEN','SECRET','PASSWORD']) and not k.startswith('VOICE_STUDIO_')}
        env.update(PYTHONUTF8='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',GRADIO_ANALYTICS_ENABLED='False',HF_HUB_OFFLINE='1')
        env['PYTHONPATH'] = os.pathsep.join([str(guard),str(repo.parent),str(repo),str(repo/'backend')])
        report = guard/'results.xml'
        code = subprocess.call([sys.executable,'-m','pytest','-q','--junitxml='+str(report),*sys.argv[1:]],env=env,cwd=repo)
        if code: return code
        xml=ET.parse(report).getroot()
        suites=[xml] if xml.tag=='testsuite' else xml.findall('testsuite')
        counts={k:sum(int(s.get(k,'0')) for s in suites) for k in ['tests','failures','errors','skipped']}
        print('OFFLINE_TEST_COUNTS',counts,flush=True)
        if counts['tests']-counts['skipped']<=0 or counts['failures'] or counts['errors']: return 1
        if os.environ.get('GITHUB_STEP_SUMMARY'):
            with open(os.environ['GITHUB_STEP_SUMMARY'],'a',encoding='utf-8') as out: out.write(f'Python tests: {counts}.\n')
    return 0
if __name__=='__main__': raise SystemExit(main())
