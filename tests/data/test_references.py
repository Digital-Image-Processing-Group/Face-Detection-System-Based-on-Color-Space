"""Reviewed mask provenance, lossless reconstruction and draft barriers."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.training.annotations import rasterize, reference_mask
from src.training.assisted_annotations import finalize
from src.training.dataset import Record, fingerprint, read_manifest, read_mask, write_json, write_manifest


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.root=Path(self.temporary.name)
        self.manifest=self.root/'manifest.csv'
        Image.new('RGB',(16,12),(180,130,95)).save(self.root/'photo.png')
        self.row=Record('hfd_fixture','photo.png','person','person','train','','','fixture',
                        'synthetic:fixture','Asian fixture','test-only','verified','missing','missing')
        write_manifest(self.manifest,[self.row])
        self.packed=np.zeros((12,16),np.uint8)
        self.packed[2:10,3:12]=3;self.packed[4:6,6:8]=2;self.packed[10:,5:10]=1
        self.reference=self.root/'annotations/reference/hfd_fixture.png'
        self.reference.parent.mkdir(parents=True);Image.fromarray(self.packed).save(self.reference)
        self.spec={'image_id':'hfd_fixture','annotator':'test-reviewer','width':16,'height':12,
                   'reference_path':'annotations/reference/hfd_fixture.png','reference_sha256':fingerprint(self.reference),
                   'review_status':'visually_reviewed','skin':{'status':'complete'},'face':{'status':'complete'}}

    def tearDown(self):
        self.temporary.cleanup()

    def test_reference_bits_restore_distinct_skin_and_face_regions(self):
        path=self.root/'records.json';write_json(path,self.spec)
        self.assertEqual(rasterize(self.manifest,path),1)
        row=read_manifest(self.manifest)[0]
        skin=read_mask(self.root/row.skin_mask_path,(12,16));face=read_mask(self.root/row.face_mask_path,(12,16))
        self.assertEqual((skin[4,6],face[4,6]),(0,1))
        self.assertEqual((skin[11,6],face[11,6]),(1,0))
        np.testing.assert_array_equal(skin,(self.packed & 1)!=0)

    def test_reference_rejects_tampering_unreviewed_status_and_invalid_values(self):
        with self.assertRaisesRegex(ValueError,'Unreviewed'):
            reference_mask(self.manifest,dict(self.spec,review_status='pending'),(12,16),'skin')
        Image.fromarray(self.packed*85).save(self.reference)
        with self.assertRaisesRegex(ValueError,'SHA256'):
            reference_mask(self.manifest,self.spec,(12,16),'skin')
        invalid=dict(self.spec,reference_sha256=fingerprint(self.reference))
        with self.assertRaisesRegex(ValueError,'encoding'):
            reference_mask(self.manifest,invalid,(12,16),'skin')

    def test_finalize_requires_an_explicit_complete_review_ledger(self):
        review=self.root/'review.json';write_json(review,{'reviewed_image_ids':[]})
        with self.assertRaisesRegex(ValueError,'every image'):
            finalize(self.manifest,review,self.root/'final.csv')
        self.assertFalse((self.root/'final.csv').exists())

    def test_finalize_freezes_reviewed_bytes_and_refuses_changed_drafts(self):
        proposals=self.root/'annotations/proposals'
        hashes={}
        for kind,bit in [('skin',1),('face',2)]:
            target=proposals/kind/'hfd_fixture.png';target.parent.mkdir(parents=True)
            Image.fromarray(((self.packed & bit)!=0).astype(np.uint8)*255).save(target)
            hashes[kind]=fingerprint(target)
        write_json(proposals/'provenance.json',{'hfd_fixture':{'source_image_sha256':fingerprint(self.root/'photo.png')}})
        review=self.root/'review.json'
        write_json(review,{'reviewed_image_ids':['hfd_fixture'],'reviewer':'codex_visual_review',
                           'reference_limitations':'synthetic test fixture, not ground truth',
                           'draft_sha256':{'hfd_fixture':hashes}})
        self.assertEqual(finalize(self.manifest,review,self.root/'final.csv')['reviewed_images'],1)
        (proposals/'skin/hfd_fixture.png').write_bytes((proposals/'face/hfd_fixture.png').read_bytes())
        with self.assertRaisesRegex(ValueError,'draft hash'):
            finalize(self.manifest,review,self.root/'changed.csv')


if __name__=='__main__':
    unittest.main()
