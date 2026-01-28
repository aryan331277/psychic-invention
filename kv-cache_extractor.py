class KVCacheExtractor:
    def __init__(self, config: ReKVConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        self.vision_model = CLIPVisionModel.from_pretrained(config.vision_model).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(config.vision_model)
        
        self.vision_model.eval()        
    
    def extract_kv_from_frames(self, frames: List[Image.Image], segment_id: int) -> str:
        with torch.no_grad():
            inputs = self.processor(images=frames, return_tensors="pt", padding=True)
            p_values = inputs['p_values'].to(self.device)
            
            if self.config.use_fp16:
            p_values= p_values.half()
            out=self.vision_model(p_values, output_hidden_states=True)#feature extraction
            kv_proxy = out.last_hidden_state.cpu()
            
            cache_path= os.path.join(self.config.kv_cache_dir, f"kv_seg_{segment_id:06d}.safetensors")
            save_file({"kv_cache": kv_proxy}, cache_path)
            
            return cache_path
    def load_kv_cache(self, cache_path: str) -> torch.Tensor:
        cached = load_file(cache_path)
        return cached["kv_cache"].to(self.device)
