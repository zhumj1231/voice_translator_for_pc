"""Switch only cloud provider settings; retain local models, proxy and budgets."""
import argparse
import json
from pathlib import Path
from paths import config_path

parser = argparse.ArgumentParser()
parser.add_argument('provider', choices=['xai', 'gemini'])
args = parser.parse_args()
path = Path(__file__).with_name('config.json')
config = json.loads(config_path().read_text(encoding='utf-8-sig'))
settings = ({'provider':'xai','endpoint':'https://api.x.ai/v1',
             'model':'grok-voice-transcribe-2.0','key_env':'XAI_API_KEY'}
            if args.provider == 'xai' else
            {'provider':'gemini_interactions','endpoint':'https://generativelanguage.googleapis.com/v1beta',
             'model':'gemini-3.5-transcribe','key_env':'GEMINI_API_KEY'})
config['cloud'].update(settings)
temporary = path.with_suffix('.json.tmp')
temporary.write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
temporary.replace(path)
print('Cloud provider saved: '+args.provider+'. Close and reopen the subtitle app to apply.')
