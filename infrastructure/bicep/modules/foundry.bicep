@description('Azure AI Foundry service name')
param name string

param location string

@description('Chat model deployment name')
param chatDeploymentName string = 'gpt-54'

@description('Embedding model deployment name')
param embeddingDeploymentName string = 'text-embedding-3-large'

resource aiServicesAccount 'Microsoft.CognitiveServices/accounts@2024-04-01-preview' = {
  name: name
  location: location
  kind: 'AIServices'
  sku: {
    name: 'S0'
  }
  properties: {
    customSubDomainName: name
    publicNetworkAccess: 'Enabled'
  }
}

resource chatDeployment 'Microsoft.CognitiveServices/accounts/deployments@2024-04-01-preview' = {
  parent: aiServicesAccount
  name: chatDeploymentName
  sku: {
    name: 'GlobalStandard'
    capacity: 40
  }
  properties: {
    model: {
      format: 'OpenAI'
      name: 'gpt-5.4'
      version: '2026-03-05'
    }
  }
}

resource embeddingDeployment 'Microsoft.CognitiveServices/accounts/deployments@2024-04-01-preview' = {
  parent: aiServicesAccount
  name: embeddingDeploymentName
  sku: {
    name: 'GlobalStandard'
    capacity: 120
  }
  properties: {
    model: {
      format: 'OpenAI'
      name: 'text-embedding-3-large'
      version: '1'
    }
  }
  dependsOn: [chatDeployment]
}

output endpoint string = aiServicesAccount.properties.endpoint
output aiServicesAccountId string = aiServicesAccount.id
