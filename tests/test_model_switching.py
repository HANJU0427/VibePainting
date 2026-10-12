import types
import unittest
import weakref
from unittest.mock import patch
from PIL import Image, ImageDraw
from inpaint_engine import ModelManager


class ModelSwitchingTests(unittest.TestCase):
    def setUp(self):
        availability=patch('inpaint_engine.require_available')
        availability.start();self.addCleanup(availability.stop)
        self.manager = ModelManager()
        self.image = Image.new('RGB', (160,80), (30,60,90))
        self.mask = Image.new('L', self.image.size)
        ImageDraw.Draw(self.mask).rectangle((60,20,100,60), fill=255)

    def test_model_switch_releases_previous_and_preserves_outside_pixels(self):
        loaded, refs, calls = [], [], []
        class Pipeline:
            def __call__(self, **kwargs):
                calls.append(kwargs)
                size = kwargs['width']
                return types.SimpleNamespace(images=[Image.new('RGB',(size,size),'red')])
        def load(model):
            self.assertTrue(all(ref() is None for ref in refs), 'Old model is still resident at next load')
            loaded.append(model)
            pipe=Pipeline();refs.append(weakref.ref(pipe));return pipe
        with patch.object(self.manager,'load',side_effect=load):
            for model in ['sd15_inpaint','sd15_inpaint','sdxl_inpaint','lightning','sd15_inpaint']:
                result=self.manager.generate(self.image,self.mask,model=model,seed=42)
                self.assertEqual(result.size,self.image.size)
                self.assertEqual(result.getpixel((80,40)),(255,0,0))
                self.assertEqual(result.getpixel((0,0)),(30,60,90))
        self.assertEqual(loaded,['sd15_inpaint','sdxl_inpaint','lightning','sd15_inpaint'])
        self.assertEqual([c['width'] for c in calls],[512,512,1024,1024,512])
        self.assertEqual([c['num_inference_steps'] for c in calls],[30,30,20,2,30])
        self.assertAlmostEqual(calls[2]['strength'],.99)
        self.assertEqual(calls[2]['image'].getpixel((512,512)),(30,60,90))
        self.assertNotIn('control_image',calls[2])
        self.assertEqual(calls[3]['control_image'].getpixel((512,512)),(255,255,255))
        self.manager.release()

    def test_failed_load_can_be_retried_without_stale_model(self):
        with patch.object(self.manager,'load',side_effect=RuntimeError('load failure')):
            with self.assertRaisesRegex(RuntimeError,'load failure'):
                self.manager.generate(self.image,self.mask,model='sd15_inpaint')
        self.assertIsNone(self.manager.pipeline)
        self.assertIsNone(self.manager.model_id)
        pipe=lambda **kw:types.SimpleNamespace(images=[Image.new('RGB',(512,512),'red')])
        with patch.object(self.manager,'load',return_value=pipe):
            self.manager.generate(self.image,self.mask,model='sd15_inpaint')
        self.assertEqual(self.manager.model_id,'sd15_inpaint')

    def test_replaced_output_is_reported_as_error(self):
        pipe=lambda **kw:types.SimpleNamespace(images=[Image.new('RGB',(512,512),'black')],nsfw_content_detected=[True])
        with patch.object(self.manager,'load',return_value=pipe):
            with self.assertRaisesRegex(ValueError,'检测器'):
                self.manager.generate(self.image,self.mask,model='sd15_inpaint')
        self.assertIsNone(self.manager.pipeline)


if __name__ == '__main__':
    unittest.main()
