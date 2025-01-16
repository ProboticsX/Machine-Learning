import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from agents.linkedin_lookup_agent import lookup
from third_parties.linkedin import scrape_linkedin_profile

def ice_break_with(name: str) -> str:
    linkedin_url = lookup(name) # uses agent to find the right LinkedIn URL
    linkedin_data = scrape_linkedin_profile(    # uses API to find the data on LinkedIn Profile
        linkedin_profile_url=linkedin_url,
        mock=True #set mock=True to fix the search
    )
    summary_template = """
            given the LinkedIn information {information} about a person from I want you to create:
            1. a short summary
            2. two interesting facts about them
        """
    summary_prompt_template = PromptTemplate(input_variables=["information"], template=summary_template)
    llm = ChatOpenAI(temperature=1, model_name="gpt-4o-mini")
    chain = summary_prompt_template | llm | StrOutputParser()

    res = chain.invoke(input={"information": linkedin_data})
    print(res)


if __name__ == "__main__":
    print("Enter Ice Breaker! :D")
    load_dotenv()
    ice_break_with("Anshika Jain Assurance Consultant Contractor @ EY GDS | Ex- GT | Ex-Acumen")
