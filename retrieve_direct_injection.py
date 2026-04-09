import json


with open('data/synthetic/flagged/direct_injection_flagged.jsonl') as f:
    for line in f:
        ex = json.loads(line)
        if 'verified_category' in ex:
            print('VERIFIED AS:', ex['verified_category'])
            print('TEXT:', ex['text'][:80])
            print()