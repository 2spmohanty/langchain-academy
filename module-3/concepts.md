1. What is the role of AgentState with add_messages — why not just use a plain list?

Add messages , give message special context of adding it as the Message Type Object.

2. What would happen if you removed the initiating_operation node and connected START directly to 
llm_assistant_node?

The AGent will not ground its calculation to the tools available.

3. In your router_node, you check last_message.tool_calls — 
what type is last_message when the tool has just finished executing and the result is back in state?

No I check the event type to know if event_type == "on_tool_end"