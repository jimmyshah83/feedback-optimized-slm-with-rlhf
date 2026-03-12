@description('Azure ML workspace name')
param name string

param location string

@description('Storage account resource ID to associate with the workspace')
param storageAccountId string

resource appInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: '${name}-insights'
  location: location
  kind: 'web'
  properties: {
    Application_Type: 'web'
  }
}

resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: '${replace(name, '-', '')}kv'
  location: location
  properties: {
    tenantId: subscription().tenantId
    sku: {
      family: 'A'
      name: 'standard'
    }
    accessPolicies: []
    enableSoftDelete: true
    softDeleteRetentionInDays: 7
  }
}

resource mlWorkspace 'Microsoft.MachineLearningServices/workspaces@2024-04-01' = {
  name: name
  location: location
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    storageAccount: storageAccountId
    keyVault: keyVault.id
    applicationInsights: appInsights.id
  }
}

output workspaceName string = mlWorkspace.name
output workspaceId string = mlWorkspace.id
