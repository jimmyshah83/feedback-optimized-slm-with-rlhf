@description('Azure AI Foundry service name')
param name string

param location string

@description('Chat model deployment name')
param chatDeploymentName string = 'gpt-54'

@description('Embedding model deployment name')
param embeddingDeploymentName string = 'text-embedding-3-large'

@description('Project name within the Foundry account')
param projectName string = 'rlaif-project'

@description('Azure AI Search service resource ID for connection')
param searchServiceId string = ''

@description('Azure AI Search endpoint for connection')
param searchEndpoint string = ''

resource aiServicesAccount 'Microsoft.CognitiveServices/accounts@2025-04-01-preview' = {
  name: name
  location: location
  kind: 'AIServices'
  identity: {
    type: 'SystemAssigned'
  }
  sku: {
    name: 'S0'
  }
  properties: {
    customSubDomainName: name
    publicNetworkAccess: 'Enabled'
    allowProjectManagement: true
    disableLocalAuth: true
  }
}

resource project 'Microsoft.CognitiveServices/accounts/projects@2025-04-01-preview' = {
  parent: aiServicesAccount
  name: projectName
  location: location
  properties: {}
}

resource searchConnection 'Microsoft.CognitiveServices/accounts/connections@2025-04-01-preview' = if (!empty(searchServiceId)) {
  parent: aiServicesAccount
  name: 'ai-search-connection'
  properties: {
    category: 'CognitiveSearch'
    target: searchEndpoint
    authType: 'AAD'
    metadata: {
      ResourceId: searchServiceId
    }
  }
}

resource chatDeployment 'Microsoft.CognitiveServices/accounts/deployments@2025-04-01-preview' = {
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

resource embeddingDeployment 'Microsoft.CognitiveServices/accounts/deployments@2025-04-01-preview' = {
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
output foundryPrincipalId string = aiServicesAccount.identity.principalId
var baseEndpoint = endsWith(aiServicesAccount.properties.endpoint, '/') ? substring(aiServicesAccount.properties.endpoint, 0, length(aiServicesAccount.properties.endpoint) - 1) : aiServicesAccount.properties.endpoint
output projectEndpoint string = '${baseEndpoint}/api/projects/${projectName}'
