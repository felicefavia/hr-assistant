import re
import numpy as np
from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import os

class SemanticChunking:

    def __init__(self, breakpoint_percentile=95, buffer_size=1):
        self.embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("AI_API_KEY"),  model="text-embedding-3-small")
        self.breakpoint_percentile = breakpoint_percentile
        self.buffer_size = buffer_size

    def _process_senteces(self, text):
        sentences = [
            {"sentence": s, "index":i}
            for i, s in enumerate(re.split(r"(?<=[.?!])\s+", text))
        ]

        for i, current in enumerate(sentences):
            context_range = range(
                max(0, i - self.buffer_size),
                min(len(sentences), i + self.buffer_size +1),
            )
            current["combined_sentence"] = " ".join(
                sentences[j]["sentence"] for j in context_range
            )

        return sentences

    def _calculate_distances(self, sentences):
        embeddings = self.embeddings.embed_documents(
            [s["combined_sentence"] for s in sentences]
        )

        distances = []
        for i in range(len(sentences) -1):
            distance = 1 - cosine_similarity(
                [embeddings[i]],
                [embeddings[i + 1]]
            )[0][0]
            distances.append(distance)

        return distances

    
    def chunk_text(self, text):
        sentences = self._process_senteces(text)
        print("SENTENCES: ", sentences[:2])
        distances = self._calculate_distances(sentences)
        print("DISTANCES:", distances[:2])

        # Determina i punti di divisione sul percentile
        threshold = np.percentile(distances, self.breakpoint_percentile)
        split_points = [i for i, d in enumerate(distances) if d > threshold]
        print("SPLIT POINTS", split_points)
        chunks = []
        start = 0

        for point in split_points + [len(sentences) -1]:
            chunk = " ".join(s["sentence"] for s in sentences[start:point+1])
            print("CHUNCK", chunk)
            chunks.append(chunk)
            start = point +1

        return chunks