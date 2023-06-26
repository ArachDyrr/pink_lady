import os

settings = {
    'host': os.environ.get('ACCOUNT_HOST', 'https://pinklady-no-sql.documents.azure.com:443/'),
    'master_key': os.environ.get('ACCOUNT_KEY', '6xGPcu9O6RvvyENR26Ijhf0hORgvHiOODn1LvNWp37oJKLO3RFt12cfFnKcSmU6QK9crmyWLucx8ACDbg7MDJA=='),
    'database_id': os.environ.get('COSMOS_DATABASE', 'own_test_zero'),
    'container_id': os.environ.get('COSMOS_CONTAINER', 'container_0'),
    'userAccountID': 'pinky' 
}