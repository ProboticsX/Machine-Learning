from common_imports import *
from graph import graph
from functions import displayGraph

class MessageEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, '__dict__'):
            # Convert the object to a dictionary
            d = obj.__dict__.copy()
            # Remove any non-serializable attributes if needed
            if 'lc_kwargs' in d:
                d['lc_kwargs'] = str(d['lc_kwargs'])
            return d
        return super().default(obj)

if __name__ == "__main__":
    print("Hello from Reflection Graph!")
    tweet = """ @LangChainAI — newly Tool Calling feature is seriously underrated. \n
                After a long wait, it's  here- making the implementation of agents across different models with function calling - super easy. \n
                Made a video covering their newest blog post"""
    displayGraph(graph)
    response = graph.invoke({"messages":[tweet]})
    
    # Store complete message objects
    messages_data = {
        "messages": response['messages'],
        "timestamp": datetime.now().isoformat()
    }
    
    # Create a unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"reflection_response_{timestamp}.json"
    
    # Save to JSON file with custom encoder
    with open(output_file, 'w') as f:
        json.dump(messages_data, f, indent=2, cls=MessageEncoder)
    
    print(f"\nResponse saved to: {output_file}")
    print("\nMessages:")
    for msg in response['messages']:
        msg.pretty_print()