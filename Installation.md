# Neo4j Local setup in k8
## Add Neo4J repo
helm repo add neo4j opens in new tabhttps://helm.neo4j.com/neo4j

## Install Neo4j using Helm
set RELEASE_NAME=graphrag

helm install neo4j neo4j/neo4j -f neo4j-custom-values.yaml

## watch the rollout status
kubectl rollout status --watch statefulset/graphrag

## Port forwarding
kubectl port-forward svc/graphrag tcp-bolt tcp-http

neo4j can be accessed using http://localhost:7474/

# Qdrant Vector Database setup in k8
## Install Qdrant using helm
helm repo add qdrant https://qdrant.to/helm

helm install qdrant qdrant/qdrant -f qdrant-custom-values.yaml

## Port forwarding for use
kubectl port-forward svc/qdrant 6333:6333 6334:6334 6335:6335

# OLLama in k8
## Add helm repo
helm repo add ollama-helm https://otwld.github.io/ollama-helm/

## Install Ollama
helm install ollama ollama-helm/ollama --values ollama-custom-values.yaml

kubectl port-forward svc/ollama 11434:11434

## Using embedding model 
curl --location 'http://localhost:11434/api/embed' \
--header 'Content-Type: application/json' \
--data '{
  "model": "bge-m3",
  "input": "Why is the sky blue?"
}'

## Using Chat Completion model
curl --location 'http://localhost:11434/api/chat' \
--header 'Content-Type: application/json' \
--data '{
  "model": "deepseek-r1:1.5b",
  "messages": [
    {
      "role": "user",
      "content": "why is the sky blue?"
    }
  ],
  "stream": false
}'
