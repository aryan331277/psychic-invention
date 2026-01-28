class MultimodalVectorDB:
    def __init__(self, config: ReKVConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        self.clip_model = CLIPModel.from_pretrained(config.vision_model).to(self.device)#initializing
        self.clip_processor = CLIPProcessor.from_pretrained(config.vision_model)
        
        if config.use_fp16:
            self.clip_model = self.clip_model.half()
        
        self.clip_model.eval()
        
        # Initialize FAISS
        self.index = faiss.IndexFlatIP(config.embedding_dim)#cosine
        self.metadata = []
        
    
    def encode_frames(self, frames):
        with torch.no_grad():
            inputs = self.clip_processor(imgs=frames, return_tensors="pt")
            p_values = inputs['p_values'].to(self.device)
            
            if self.config.use_fp16:
                p_values = p_values.half()
            
            # Get image embeddings
            img_features=self.clip_model.get_img_features(p_values)
            
            embedding = img_features.mean(dim=0).numpy()
            embedding =embedding/np.linalg.norm(embedding)
            
            return embedding.astype('float32')
    
    def encode_text(self, query):
        with torch.no_grad():
            text_inputs=self.clip_processor(text=[query], return_tensors="pt", padding=True)
            text_inputs={k: v.to(self.device) for k, v in text_inputs.items()}
            
            text_features =self.clip_model.get_text_features(**text_inputs)
            text_features = F.normalize(text_features, dim=-1)
            
            # Convert to numpy
            embedding=text_features.numpy().flatten()
        
        return embedding.astype('float32')
    
    def add_segment(self,embedding,metadata):
        embedding = embedding.reshape(1, -1)
        self.index.add(embedding)
        self.metadata.append(metadata)
    
    def search(self, q_embedding, k= 5):
        q_embedding = q_embedding.reshape(1, -1)
        
        scores, indices = self.index.search(q_embedding, k)#faiss search
      
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.metadata):
                results.append((idx, float(score), self.metadata[idx]))
        return results
    
    def save(self):
        os.makedirs(self.config.vector_db_path, exist_ok=True)
        
        # Save FAISS index
        index_path = os.path.join(self.config.vector_db_path, "index.faiss")
        faiss.write_index(self.index, index_path)
        
        with open(self.config.metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        
    
    def load(self):
        index_path = os.path.join(self.config.vector_db_path, "index.faiss")
        
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            
            with open(self.config.metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
            return True
        return False
