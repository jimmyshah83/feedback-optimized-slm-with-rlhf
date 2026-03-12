targetScope = 'subscription'

@description('Base name prefix for all resources')
param baseName string

@description('Azure region for resource deployment')
param location string = 'eastus2'

@description('Chat model deployment name')
param chatDeploymentName string = 'gpt-54'

@description('Embedding model deployment name')
param embeddingDeploymentName string = 'text-embedding-3-large'

resource rg 'Microsoft.Resources/resourceGroups@2024-03-01' = {
  name: '${baseName}-rg'
  location: location
}

module storage 'modules/storage.bicep' = {
  scope: rg
  name: 'storage'
  params: {
    name: '${replace(baseName, '-', '')}store'
    location: location
  }
}

module search 'modules/ai_search.bicep' = {
  scope: rg
  name: 'search'
  params: {
    name: '${baseName}-search'
    location: location
  }
}

module foundry 'modules/foundry.bicep' = {
  scope: rg
  name: 'foundry'
  params: {
    name: '${baseName}-ai'
    location: location
    chatDeploymentName: chatDeploymentName
    embeddingDeploymentName: embeddingDeploymentName
  }
}

module cosmos 'modules/cosmos.bicep' = {
  scope: rg
  name: 'cosmos'
  params: {
    name: '${baseName}-cosmos'
    location: location
  }
}

module mlWorkspace 'modules/ml_workspace.bicep' = {
  scope: rg
  name: 'mlWorkspace'
  params: {
    name: '${baseName}-ml'
    location: location
    storageAccountId: storage.outputs.storageAccountId
  }
}

output resourceGroupName string = rg.name
output searchEndpoint string = search.outputs.endpoint
output foundryEndpoint string = foundry.outputs.endpoint
output cosmosEndpoint string = cosmos.outputs.endpoint
output storageAccountName string = storage.outputs.storageAccountName
output mlWorkspaceName string = mlWorkspace.outputs.workspaceName
