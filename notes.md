# Image Handling
- Try Multimodel retriver. **MultiModalVectorIndexRetriever** retrieve relevant images from PDF for a question and send it to GPT4V tp respond.
- Use OCR during load time to convert image to text and load it as a separate document with relation to the original document in knowledge graph

# Triplet to get semantic meaning
- Try chaning the default Prompt to give context 
- Add context based Examples in Triplet formation
- Add detailed relationship summary
- have a second order Knowledge graph with the description of all triplets extracted from chunks. 
- varying hyper parameters e.g. number of triplets per chunk

