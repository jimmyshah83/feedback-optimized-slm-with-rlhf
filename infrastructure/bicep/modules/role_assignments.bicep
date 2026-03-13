@description('Name of the Azure AI Search service')
param searchServiceName string

@description('Principal ID of the Foundry system-assigned managed identity')
param foundryPrincipalId string

resource searchService 'Microsoft.Search/searchServices@2024-03-01-preview' existing = {
  name: searchServiceName
}

// Search Index Data Reader — allows the Foundry agent to query the search index
resource foundrySearchReader 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(searchService.id, foundryPrincipalId, 'search-index-data-reader')
  scope: searchService
  properties: {
    roleDefinitionId: subscriptionResourceId(
      'Microsoft.Authorization/roleDefinitions',
      '1407120a-92aa-4202-b7e9-c0e197c71c8f'
    )
    principalId: foundryPrincipalId
    principalType: 'ServicePrincipal'
  }
}

// Search Service Contributor — allows the Foundry to manage search resources
resource foundrySearchContributor 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(searchService.id, foundryPrincipalId, 'search-service-contributor')
  scope: searchService
  properties: {
    roleDefinitionId: subscriptionResourceId(
      'Microsoft.Authorization/roleDefinitions',
      '7ca78c08-252a-4471-8644-bb5ff32d4ba0'
    )
    principalId: foundryPrincipalId
    principalType: 'ServicePrincipal'
  }
}
