class RetrievalReasoning:

    def __init__(self, config: ReKVConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Load vector DB
        self.vector_db = MultimodalVectorDB(config)
        loaded = self.vector_db.load()
        
        # Load KV extractor (for loading caches)
        self.kv_extractor = KVCacheExtractor(config)
        
        self.clip_model = CLIPModel.from_pretrained(config.vision_model).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(config.vision_model)
        
        if config.use_fp16:
            self.clip_model = self.clip_model.half()
        
        self.clip_model.eval()
        

        self.vision_hidden_dim = 768  # CLIP ViT-B/32 vision hidden size
        self.text_hidden_dim = 512    # CLIP text projection size
        
        # Create projection layer to align dimensions
        self.vision_projector = nn.Linear(self.vision_hidden_dim, self.text_hidden_dim).to(self.device)
        if config.use_fp16:
            self.vision_projector = self.vision_projector.half()
        
    
    def retrieve_segments_with_context(self, query):
        # Encode query
        q_embedding = self.vector_db.encode_text(query)
        # Initial search
        results = self.vector_db.search(q_embedding, k=self.config.top_k_retrieval)
        
        # Expand with temporal context
        context_half_window = self.config.context_window_seconds / 2
        expanded_results = {}
        
        for seg_id, score, metadata in results:
            expanded_results[seg_id] = {
                'metadata': metadata,
                'score': score,
                'is_primary': True
            }
            
            center_time = (metadata['start_time'] + metadata['end_time']) / 2
            
            for other_meta in self.vector_db.metadata:
                other_time = (other_meta['start_time'] + other_meta['end_time']) / 2
                time_diff = abs(other_time - center_time)
                
                if time_diff <= context_half_window and other_meta['segment_id'] not in expanded_results:
                    expanded_results[other_meta['segment_id']] = {
                        'metadata': other_meta,
                        'score': score * 0.7,  # Lower score for context segments
                        'is_primary': False
                    }
        
        sorted_results = sorted(
            expanded_results.values(), 
            key=lambda x: x['metadata']['start_time']
        )
        
        return sorted_results
    
    def analyze_query(self,query,verbos,save_visuals):
        if verbose:
            print(f"\n{'='*60}")
            print(f"QUERY: {query}")
            print(f"{'='*60}\n")
        
        start_time = time.time()
        
        # Step 1: Retrieve relevant segments
        if verbose:
            print("Retrieving relevant segments...")
        
        retrieved_segments = self.retrieve_segments_with_context(query)
        
        retrieval_time = time.time() - start_time
        
        if verbose:
            print(f"Found {len(retrieved_segments)} segments (including context)")
            print(f"Retrieval time: {retrieval_time:.3f}s\n")
        
        # Step 2: Load KV caches (this is the ReKV innovation!)
        if verbose:
            print("Loading pre-computed KV caches from disk...")
        
        kv_load_start = time.time()
        loaded_features = []
        
        for seg_info in retrieved_segments:
            metadata = seg_info['metadata']
            kv_cache_path = metadata['kv_cache_path']
            
            # Load KV cache from disk
            kv_cache = self.kv_extractor.load_kv_cache(kv_cache_path)
            
            loaded_features.append({
                'kv_cache': kv_cache,
                'metadata': metadata,
                'score': seg_info['score'],
                'is_primary': seg_info['is_primary']
            })
        
        kv_load_time = time.time() - kv_load_start
        
        if verbose:
            print(f"Loaded {len(loaded_features)} KV caches")
            print(f"KV load time: {kv_load_time:.3f}s")
            print(f"✓ Total I/O time: {kv_load_time:.3f}s (vs. re-encoding: ~{len(loaded_features) * 0.5:.1f}s)")
        
        # Step 3: Reasoning over injected KV caches
        # ASSUMPTION: In true ReKV, we'd inject these into decoder's cross-attention
        # Here we simulate by computing similarity between query and cached features
        if verbose:
            print("\nReasoning over visual features...")
        
        reasoning_start = time.time()
        
        # Encode query text
        with torch.no_grad():
            text_inputs = self.clip_processor(text=[query], return_tensors="pt", padding=True)
            text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
            text_features = self.clip_model.get_text_features(**text_inputs)
            text_features = F.normalize(text_features, dim=-1)
        
        # Compute relevance scores by comparing with KV caches
        results = []
        for feat_info in loaded_features:
            kv_cache = feat_info['kv_cache']
            
            # Aggregate KV cache (mean pooling over sequence)
            # Shape: [num_frames, seq_len, hidden_dim] -> [hidden_dim]
            visual_features = kv_cache.mean(dim=[0, 1]).to(self.device)
            
            if self.config.use_fp16:
                visual_features = visual_features.half()
            
            # Project vision features to text dimension
            visual_features_proj = self.vision_projector(visual_features.unsqueeze(0))
            visual_features_proj = F.normalize(visual_features_proj, dim=-1)
            
            # Compute similarity
            with torch.no_grad():
                similarity = (text_features * visual_features_proj).sum().item()
            
            results.append({
                'segment_id': feat_info['metadata']['segment_id'],
                'video_id': feat_info['metadata'].get('video_id', 'unknown'),
                'video_path': feat_info['metadata'].get('video_path', ''),
                'start_time': feat_info['metadata']['start_time'],
                'end_time': feat_info['metadata']['end_time'],
                'retrieval_score': feat_info['score'],
                'reasoning_score': similarity,
                'combined_score': feat_info['score'] * 0.5 + similarity * 0.5,
                'is_primary_match': feat_info['is_primary']
            })
        
        # Sort by combined score
        results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        reasoning_time = time.time() - reasoning_start
        total_time = time.time() - start_time
        
        # Generate answer
        top_match = results[0] if results else None
        
        answer = {
            'query': query,
            'top_match': top_match,
            'all_matches': results[:10],  # Top 10
            'timing': {
                'retrieval': retrieval_time,
                'kv_loading': kv_load_time,
                'reasoning': reasoning_time,
                'total': total_time
            },
            'segments_analyzed': len(results)
        }
        
       
        return answer
    
   
                
