from langchain.llms import OpenAI
from langchain.chains import LLMChain, PromptTemplate
from rag_pipeline import load_retriever

retriever = load_retriever()

job_description = """
We are hiring a Frontend Developer with experience in React, JavaScript, and UI/UX design. Strong problem-solving skills and attention to detail are a must.
"""

results = retriever.get_relevant_documents(job_description)

prompt_template = """
Given the job description:
{job_description}

And the following SHL catalog entries:
{retrieved_entries}

Recommend the most relevant SHL assessments with reasons.
"""

llm = OpenAI(temperature=0.2)
chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        input_variables=["job_description", "retrieved_entries"],
        template=prompt_template
    )
)

response = chain.run(
    job_description=job_description,
    retrieved_entries="\n\n".join([doc.page_content for doc in results])
)

print(response)
