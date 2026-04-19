"""Quick single-ticket smoke test."""
import asyncio
import json

from dotenv import load_dotenv
load_dotenv(override=True)

from src.agent import configure_gemini, run_agent


async def main():
    client = configure_gemini()
    tickets = json.loads(open("data/tickets.json").read())
    ticket = tickets[0]
    print(f"Testing ticket: {ticket['ticket_id']} - {ticket['subject']}")
    print(f"Expected: {ticket['expected_action']}")
    print("---")
    
    result = await run_agent(ticket, client)
    
    print(f"Status: {result.status}")
    print(f"Final action: {result.final_action}")
    print(f"Confidence: {result.confidence_score}")
    print(f"Tools called: {result.tool_call_count}")
    print(f"Error: {result.error}")
    print(f"Processing time: {result.processing_time_ms}ms")
    
    if result.reasoning_trace:
        print(f"\nLast reasoning:")
        print(result.reasoning_trace[-1][:300])


if __name__ == "__main__":
    asyncio.run(main())
