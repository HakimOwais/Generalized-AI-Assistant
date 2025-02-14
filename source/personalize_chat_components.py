from database_setup import vector_store_transactions

def qa_transactions_driver(email, query):
    filter_condition = {"email": email}

    # Retrieve recommended documents using similarity search with the email filter applied
    results = vector_store_transactions.similarity_search(
        query, 
        k=3, 
        pre_filter={"email": {"$eq": email}}
    )
    
    # Print the retrieved documents to check the context
    print("Retrieved Documents:")
    for result in results:
        print(result)  # Print the individual results

    # Convert the results to the format expected by the chain (if needed, adjust structure)
    context = "\n".join([str(result) for result in results])  # Combine the retrieved documents into a single contex
    return context