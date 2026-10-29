"""Inspect author demographic annotations before collecting real scene photos."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import quote

from .dataset import write_json
from .public_sources import fetch

COMMIT='92b0df27ac7cb7b203317d7c9f88a9afbed18984'
BASE=f'https://raw.githubusercontent.com/bronval/Wider-Fair-Dataset/{COMMIT}/'
ANNOTATIONS={'train_split.csv':'34a57656ca670535d955f5760feedd372b0d135b',
             'validation_set.csv':'90f53075879444dfd1ae754f377a52981014bdbf',
             'test_set.csv':'915d78e17ce983ae01a7973ea43205b514abad04'}


def git_blob_sha(content: bytes) -> str:
    return hashlib.sha1(f'blob {len(content)}\0'.encode()+content).hexdigest()


def inspect(root: Path) -> None:
    grouped=defaultdict(list)
    for filename,expected in ANNOTATIONS.items():
        content=fetch(BASE+'dataset/'+filename,root/filename)
        if git_blob_sha(content)!=expected:
            raise ValueError('Author annotation Git blob mismatch')
        for row in csv.DictReader(io.StringIO(content.decode('utf-8-sig'))):
            row['author_split']=filename
            grouped[row['filename']].append(row)
    eligible={name:rows for name,rows in grouped.items()
              if all(r['Ethnicity']=='Asian' and r['Valid']=='True' for r in rows)
              and all(float(r['x2'])-float(r['x1'])>=55 and float(r['y2'])-float(r['y1'])>=65 for r in rows)}
    write_json(root/'all_asian_candidates.json',eligible)
    write_json(root/'metadata.json',{'author_repository':'https://github.com/bronval/Wider-Fair-Dataset',
        'commit':COMMIT,'annotation_git_blobs':ANNOTATIONS,'label_basis':'Single author human annotator: perceived Asian ethnicity; not participant self-report',
        'safety':'Candidate status only. Small faces were filtered by authors: compare original WIDER boxes and visually reject any unlabelled visible people before admitting a full image.',
        'candidate_images':len(eligible),'candidate_events':dict(Counter(n.split('/')[0] for n in eligible))})
    print(json.dumps({'images':len(eligible),'multi_face':sum(len(r)>1 for r in eligible.values()),
                      'events':dict(Counter(n.split('/')[0] for n in eligible))},indent=2),flush=True)


def hfd_folders(root: Path) -> None:
    result=json.loads(fetch('https://api.osf.io/v2/nodes/fkn7y/files/osfstorage/',root/'hfd_root.json'))
    print(json.dumps([{'name':r['attributes']['name'],'kind':r['attributes']['kind'],'url':r['relationships'].get('files',{}).get('links',{})} for r in result['data']],indent=2),flush=True)


def hfd_reserve(root: Path) -> None:
    import re,random
    from concurrent.futures import ThreadPoolExecutor
    from .dataset import Record,write_manifest
    from .public_sources import checked
    roots=json.loads((root/'hfd_root.json').read_text())['data']
    groups=defaultdict(dict)
    for folder in roots:
        category=folder['attributes']['name']
        if category not in ('Asian faces','East Asian faces','Southeast Asian faces'):continue
        url=folder['relationships']['files']['links']['related']['href']+'?page[size]=100'
        page=0
        while url:
            result=json.loads(fetch(url,root/f'hfd_{category}_{page}.json'))
            for f in result['data']:
                m=re.fullmatch(r'(\d+)_([A-Z]+)([23678])\.jpg',f['attributes']['name'])
                if m:
                    code=f'{m[2]}_{int(m[1]):03}';f['category']=category
                    groups[code][m[3]]=f
            url=result.get('links',{}).get('next');page+=1
    old=set(json.loads(Path('data/sources/hawaii_selection.json').read_text())['selected_participants'])
    eligible=sorted(k for k,v in groups.items() if k not in old and all(c in v for c in ('6','7','8')))
    random.Random(20261029).shuffle(eligible);chosen=eligible[:8]
    if len(chosen)!=8:raise ValueError('Need eight unused author participants')
    def download(pair):
        code,variant=pair;f=groups[code][variant];name=f['attributes']['name'];target=Path('data/images/hawaii')/name
        content=fetch(f['links']['download'],target);sha=f['attributes']['extra']['hashes']['sha256'];checked(content,sha)
        return {'image_id':'hfd_'+Path(name).stem,'author_participant_code':code,'author_filename':name,
            'source_group':f['category'],'variant':variant,'source_url':'https://osf.io/fkn7y/',
            'author_page':'https://www.diversitysocialcoglab.com/resources','download_url':f['links']['download'],
            'osf_file_id':f['id'],'sha256':sha,'author_sha256':sha,'terms':'HFD education/research/example stimuli only; no raw photo redistribution'}
    with ThreadPoolExecutor(max_workers=6) as pool: sources=list(pool.map(download,[(c,v) for c in chosen for v in ('6','7','8')]))
    rows=[Record(s['image_id'],'images/hawaii/'+s['author_filename'],'hfd_'+s['author_participant_code'],
        'hfd_'+s['author_participant_code'],'test','','','controlled_portrait;fresh_holdout;'+('frontal' if s['variant']=='6' else 'profile'),
        s['source_url'],s['source_group'],'HFD-education-research-example-stimuli-only','verified','missing','missing') for s in sources]
    write_manifest(Path('data/reserve.manifest.csv'),rows)
    write_json(Path('data/sources/hawaii_reserve.json'),{'seed':20261029,'selected_participants':chosen,'records':sources,'purpose':'Untouched final participant holdout; chosen before observing model predictions or metrics.'})
    print(json.dumps({'fresh_participants':chosen,'photos':len(rows)}),flush=True)


def norming(root: Path) -> None:
    import zipfile,re,xml.etree.ElementTree as E
    from .public_sources import checked
    data=json.loads((root/'hfd_root.json').read_text())['data']
    source=next(f for f in data if f['attributes']['name']=='HFD Norming Data.xlsx')
    raw=fetch(source['links']['download'],root/'norming.xlsx');checked(raw,source['attributes']['extra']['hashes']['sha256'])
    ns={'m':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        strings=[''.join(t.itertext()) for t in E.fromstring(z.read('xl/sharedStrings.xml')).findall('m:si',ns)]
        lookup={}
        for row in E.fromstring(z.read('xl/worksheets/sheet1.xml')).findall('m:sheetData/m:row',ns)[1:]:
            cells={}
            for c in row.findall('m:c',ns):
                v=c.find('m:v',ns);text=v.text if v is not None else ''
                if c.get('t')=='s':text=strings[int(text)]
                cells[re.sub('[0-9]','',c.get('r'))]=text
            if cells.get('A'):lookup[cells['A'].strip()]=cells
    path=Path('data/sources/hawaii_participants.json');doc=json.loads(path.read_text(encoding='utf-8'))
    existing={p['participant_code'] for p in doc['participants']}
    for s in json.loads(Path('data/sources/hawaii_reserve.json').read_text())['records']:
        if s['variant']!='6' or s['author_participant_code'] in existing:continue
        c=lookup[Path(s['author_filename']).stem]
        doc['participants'].append(dict(participant_code=s['author_participant_code'],neutral_filename=s['author_filename'],researcher_categorized_race=c['B'],self_reported_race=c['C']))
    write_json(path,doc);print('Author metadata participants:',len(doc['participants']),flush=True)


def candidates(root: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor
    import random
    all_rows=json.loads((root/'all_asian_candidates.json').read_text())
    rng=random.Random(20261029)
    selected=[]
    for category,count in [('1--Handshaking',12),('12--Group',8),('19--Couple',5),('20--Family_Group',2),
                           ('18--Concerts',6),('4--Dancing',6),('50--Celebration_Or_Party',4),
                           ('13--Interview',5),('27--Spa',4),('49--Greeting',5),('9--Press_Conference',4),
                           ('52--Photographers',4),('31--Waiter_Waitress',5)]:
        names=sorted(n for n in all_rows if n.startswith(category+'/'))
        rng.shuffle(names); selected.extend(names[:count])
    def download(name):
        key='dataset/images/'+name; target=root/'candidates'/Path(name).name
        payload=fetch(BASE+quote(key),target,5_000_000)
        return {'filename':name,'file':str(target),'retrieved_git_blob':git_blob_sha(payload),
                'sha256':hashlib.sha256(payload).hexdigest(),'download_url':BASE+quote(key),
                'author_annotations':all_rows[name]}
    with ThreadPoolExecutor(max_workers=6) as pool:
        records=[]
        for record in pool.map(download,selected):
            records.append(record)
            if len(records)%10==0:print(f'Pinned-author-source scene candidates: {len(records)}/{len(selected)}',flush=True)
    write_json(root/'downloaded_candidates.json',records)
    print(json.dumps({'downloaded_candidates':len(records)}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['inspect','candidates','hfd-folders','hfd-reserve','norming'])
    parser.add_argument('--directory',type=Path,default=Path('data/cache/scene_review'))
    args=parser.parse_args()
    {'inspect':inspect,'candidates':candidates,'hfd-folders':hfd_folders,'hfd-reserve':hfd_reserve,'norming':norming}[args.command](args.directory)
