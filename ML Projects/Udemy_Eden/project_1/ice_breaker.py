from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from project_1.agents.linkedin_lookup_agent import lookup
from project_1.agents.twitter_lookup_agent import lookup_twitter
from output_parsers import summary_parser
from project_1.third_parties.linkedin import scrape_linkedin_profile
from project_1.third_parties.twitter import scrape_user_tweets


def ice_break_with(name: str) -> str:
    linkedin_url = lookup(name) # uses agent to find the right LinkedIn URL
    linkedin_data = scrape_linkedin_profile(    # uses API to find the data on LinkedIn Profile
        linkedin_profile_url=linkedin_url,
        mock=False #set mock=True to fix the search
    )
    twitter_username = lookup_twitter(name) # uses agent to find the right Twiiter Username
    twitter_data = scrape_user_tweets(    # uses API to find the tweets on Twitter Profile
        username=twitter_username,
        mock=False #set mock=True to fix the search
    )
    summary_template = """
            given the LinkedIn information {information} and twitter posts {twitter_posts} about a person from I want you to create:
            1. a short summary
            2. two interesting facts about them
            Use the data from Linkedin and Twitter
            \n{format_instructions}
        """
    summary_prompt_template = PromptTemplate(
                                            input_variables=["information", "twitter_posts"],
                                            partial_variables={"format_instructions":summary_parser.get_format_instructions()},
                                            template=summary_template
                                            )
    llm = ChatOpenAI(temperature=1, model_name="gpt-4o-mini")
    # chain = summary_prompt_template | llm | StrOutputParser()
    chain = summary_prompt_template | llm | summary_parser
    res = chain.invoke(input={"information": linkedin_data, "twitter_posts": twitter_data})
    print("Data about the user:")
    print(res)

if __name__ == "__main__":
    print("Enter Ice Breaker! :D")
    load_dotenv()
    ice_break_with("Sundar Pichai Google CEO")
