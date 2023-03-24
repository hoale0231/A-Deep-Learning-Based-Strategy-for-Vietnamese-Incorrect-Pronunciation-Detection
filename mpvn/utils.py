import os
import json
import torch
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from pathlib import Path
from typing import List, Dict
from pydub import AudioSegment
from collections import defaultdict
from textgrid import TextGrid, IntervalTier

def average_checkpoints(model: nn.modules, filenames: List[Path], device: torch.device = torch.device("cpu")) -> dict:
    n = len(filenames)

    avg = torch.load(filenames[0], map_location=device)['state_dict']

    # Identify shared parameters. Two parameters are said to be shared
    # if they have the same data_ptr
    uniqued: Dict[int, str] = dict()

    for k, v in avg.items():
        v_data_ptr = v.data_ptr()
        if v_data_ptr in uniqued:
            continue
        uniqued[v_data_ptr] = k

    uniqued_names = list(uniqued.values())

    for i in range(1, n):
        state_dict = torch.load(filenames[i], map_location=device)['state_dict']
        for k in uniqued_names:
            avg[k] += state_dict[k]

    for k in uniqued_names:
        if avg[k].is_floating_point():
            avg[k] /= n
        else:
            avg[k] //= n
            
    model.load_state_dict(avg)
    
    return model

def textgrid_to_json(textgrid):
    return {
        tier.name: [
            {
                'value': interval.mark,
                'start': interval.minTime,
                'end': interval.maxTime
            }
            for interval in tier if interval.mark
        ]
        for tier in textgrid
    }
    
def list_interval_to_tier(list_interval, name):
    tier = IntervalTier(name, maxTime=list_interval[-1]['end'])
    for e in list_interval:
        tier.add(e['start'], e['end'], e['value'])
    return tier        
    
def combine_all_tier(data):
    return [
        {
            'word': word['value'],
            'score': score['value'],
            'start': word['start'],
            'end': word['end'],
            'phones': [
                {
                    'phone': phone['value'],
                    'start': phone['start'],
                    'end': phone['end']
                }
                for phone in filter(
                    lambda x: x['start'] >= word['start'] and x['end'] <= word['end'], 
                    data['phones']
                )
            ]
        }
        for word, score in zip(data['words'], data['scores'])
    ]

VOWELS = """iə iə2 iəɜ iə4 iə5 iə6
            iɛ iɛ1 iɛ2 iɛɜ iɛ4 iɛ5 iɛ6
            i i2 iɜ i4 i5 i6
            e e1 e2 eɜ e4 e5 e6 e7
            ɛ ɛ2 ɛɜ ɛ4 ɛ5 ɛ6
            yə yə2 yəɜ yə4 yə5 yə6
            y y2 yɜ y4 y5 y6
            əː əː2 əːɜ əː4 əː5 əː6
            ə ə1 ə2 əɜ ə4 ə5 ə6
            aː aː2 aːɜ aː4 aː5 aː6
            a a2 aɜ a4 a5 a6
            uə uə2 uəɜ uə4 uə5 uə6
            u u2 uɜ u4 u5 u6
            o o2 oɜ o4 o5 o6
            ɔ ɔ2 ɔɜ ɔ4 ɔ5 ɔ6
            əɪ əɪ2 əɪɜ əɪ4 əɪ5 əɪ6""".split()
                            
def convert_phone(words, phone_map):
    for word in words:
        new_phones = phone_map[word['word']]         
        if len(new_phones) == len(word['phones']):
            for old, new_phone in zip(word['phones'], new_phones):
                old['phone'] = new_phone
        elif new_phones[0] in VOWELS and len(new_phones) + 1 == len(word['phones']):
            word['phones'][1]['start'] = word['phones'][0]['start'] 
            word['phones'].pop(0)
            for old, new_phone in zip(word['phones'], new_phones):
                old['phone'] = new_phone
        elif 'əɪ' in ' '.join(new_phones):
            if len(new_phones) == 1 and len(word['phones']) == 3:
                word['phones'] = [{'phone':new_phones[0], 'start':word['start'], 'end':word['end']}]
            elif len(new_phones) == 2 and len(word['phones']) == 3:
                word['phones'][0]['phone'] = new_phones[0]
                word['phones'][1] = {
                    'phone': new_phones[1],
                    'start': word['phones'][1]['start'],
                    'end': word['phones'][2]['end']
                }
            else:
                print('something wrong', new_phones, word)
                raise
        else:
            print('diff')
            print(word, new_phones)
    return words
                
def cuts_words(utt, words):
    wav_path = f'corpus/{utt}.wav'
    sound = AudioSegment.from_file(wav_path)
    save_dir = f'cuts/{utt}'
    
    for i, word in enumerate(words):
        save_phone_dir = f"{save_dir}/{i}/phones"
        os.makedirs(save_phone_dir, exist_ok=True)
        save_path = f"{save_dir}/{i}/{i}.wav"
        word['path'] = save_path

        word_segment = sound[int(word['start']*1000):int(word['end']*1000)]
        word_segment.export(save_path, format='wav')
        
        for j, phone in enumerate(word['phones']):
            phone_segment = sound[int(phone['start']*1000):int(phone['end']*1000)]
            save_phone_path = f"{save_phone_dir}/{j}.wav"
            phone_segment.export(save_phone_path, format='wav')
            phone['path'] = save_phone_path
    return words

def get_word_and_phone_dicts(data):
    words_dict = defaultdict(list)
    phones_dict = defaultdict(list)
    for _, utt in data.items():
        for word in utt:
            words_dict[word['word']].append(word['path'])
            for phone in word['phones']:
                phones_dict[phone['phone']].append(phone['path'])
    return words_dict, phones_dict
        
    
if __name__ == '__main__':
    print('Loading meta file')
    meta_file = '/media/wicii/DDH/class/graduation_project/mpvn/Data/label_train.csv'
    df = pd.read_csv(meta_file, index_col=0)
    
    print('Loading phone map file')
    phone_map_file = '/media/wicii/DDH/class/graduation_project/mpvn/Data/phone_map.json'
    phone_map = json.load(open(phone_map_file))
    phone_map = {
        k: v.split('-')
        for k, v in phone_map.items()
    }

    all_utt = dict()
    
    print('Processing...')

    for utt, (_, _, text, _, score) in tqdm(df.iterrows(), total=len(df)):
        # Validate data
        spk_id = utt.split('_')[0]
        score = score.strip().split()
        text = text.strip().split()
        assert len(score) ==  len(text), 'num scores and words mismatch'
        
        # Read file
        textgrid_path = f'corpus/{spk_id}/{utt}.TextGrid' 
        textgrid = TextGrid()
        textgrid.read(textgrid_path)
        
        data = textgrid_to_json(textgrid)
        assert len(data['words']) == len(score), 'num words mismatch'
        
        # Add score if not yet
        if 'scores' in data:
            assert len(data['scores']) == len(score), 'num scores mis match'
        else:
            data['scores'] = [
                {
                    'value': score,
                    'start': word['start'],
                    'end': word['end']
                }
                for score, word in zip(score, data['words'])
            ]
            textgrid.append(list_interval_to_tier(data['scores'],  'scores'))
            textgrid.write(open(textgrid_path, 'w'))
        
        # Combine all tier
        combined = combine_all_tier(data)
        
        # Filter correct & valid pronunciation word
        filtered = list(filter(
            lambda word: 
                (int(word['score']) == 1) 
                and ('spn' not in [i['phone'] for i in word['phones']])
                and word['word'] != 'quốc', 
            combined
        ))

        # Convert phone format
        converted = convert_phone(filtered, phone_map)
        cuts = cuts_words(f'{spk_id}/{utt}', converted)
        all_utt[utt] = cuts
    
    print('Save result')
    
    json.dump(all_utt, open('alinged.json', 'w'), ensure_ascii=False)
    # all_utt = json.load(open('alinged.json'))
    words_dict, phones_dict = get_word_and_phone_dicts(all_utt)
    json.dump(words_dict, open('words_dict.json', 'w'), ensure_ascii=False)
    json.dump(phones_dict, open('phones_dict.json', 'w'), ensure_ascii=False)
