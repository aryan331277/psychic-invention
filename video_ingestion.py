class VideoIngestionPipeline:
    
    def __init__(self, config: ReKVConfig):
        self.config = config
        self.kv_extractor = KVCacheExtractor(config)
        self.vector_db = MultimodalVectorDB(config)
        
    def extract_frames_from_video(self,video_path):

        cap = cv2.VideoCapture(video_path)
        
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        frames = []
        frame_interval = max(1, int(fps / self.config.fps_sample))
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = img.fromarray(frame_rgb)
                timestamp = frame_count / fps
                frames.append((timestamp, pil_frame))
            
            frame_count += 1
        
        cap.release()
        
        return frames
    
    def ingest_video(self,video_path,video_id):
        if video_id is None:
            video_id = os.path.basename(video_path)
        

        frames = self.extract_frames_from_video(video_path)
        
        segments = []
        current_segment = []
        current_start_time = 0.0
        
        for timestamp, frame in frames:
            if len(current_segment) == 0:
                current_start_time = timestamp
            
            current_segment.append(frame)
            
            # Create segment when we hit the duration or max frames
            if (timestamp - current_start_time >= self.config.segment_duration or 
                len(current_segment) >= self.config.max_frames_per_segment):
                
                segments.append({
                    'frames': current_segment[:],
                    'start_time': current_start_time,
                    'end_time': timestamp
                })
                current_segment = []
        
        # Add last segment if exists
        if current_segment:
            segments.append({
                'frames': current_segment,
                'start_time': current_start_time,
                'end_time': frames[-1][0]
            })
        
        
        # Process segments
        current_total_segs = len(self.vector_db.metadata)
        
        for local_seg_id, segment in enumerate(tqdm(segments, desc=video_id[:30])):
            # Global segment ID
            global_seg_id = current_total_segs + local_seg_id
            
            # Path A: Create search embedding
            embedding = self.vector_db.encode_frames(segment['frames'])
            
            # Path B: Extract KV cache
            kv_cache_path = self.kv_extractor.extract_kv_from_frames(
                segment['frames'], 
                global_seg_id
            )
            
            # Store in vector DB with metadata
            metadata = {
                'segment_id': global_seg_id,
                'video_id': video_id,
                'video_path': video_path,
                'start_time': segment['start_time'],
                'end_time': segment['end_time'],
                'kv_cache_path': kv_cache_path,
                'num_frames': len(segment['frames'])
            }
            
            self.vector_db.add_segment(embedding, metadata)
            
            # Memory management
            if local_seg_id % 10 == 0:
                torch.cuda.empty_cache()
                gc.collect()
        
        return True
    
    def ingest_multiple_videos(self, video_paths):
        successful = 0
        failed = 0
        
        for i, video_path in enumerate(video_paths, 1):
            video_id = os.path.basename(video_path)
            
            if self.ingest_video(video_path, video_id):
                successful=successful+ 1
            else:
                failed=failed+ 1
            
            if i % 10 == 0:
                self.vector_db.save()
        
        # Final save
        self.vector_db.save()
