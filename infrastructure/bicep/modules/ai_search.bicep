@description('Azure AI Search service name')
param name string

param location string

resource searchService 'Microsoft.Search/searchServices@2024-03-01-preview' = {
  name: name
  location: location
  sku: {
    name: 'standard'
  }
  properties: {
    replicaCount: 1
    partitionCount: 1
    hostingMode: 'default'
    semanticSearch: 'standard'
  }
}

output endpoint string = 'https://${searchService.name}.search.windows.net'
output searchServiceId string = searchService.id
