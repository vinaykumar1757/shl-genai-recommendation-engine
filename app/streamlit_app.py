import streamlit as st
from scripts.rag_pipeline import load_retriever
from langchain.llms import OpenAI
from langchain.chains import LLMChain, PromptTemplate

retriever = load_retriever()
llm = OpenAI(temperature=0.2)

prompt_template = """
Given the job description:
{job_description}

And the following SHL catalog entries:
{retrieved_entries}

Recommend the most relevant SHL assessments with reasons.
"""

chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(
        input_variables=["job_description", "retrieved_entries"],
        template=prompt_template
    )
)

st.title("🔍 SHL Assessment Recommendation Engine")

job_desc = st.text_area("Paste Job Description")

if st.button("Recommend"):
    docs = retriever.get_relevant_documents(job_desc)
    results = "\n\n".join([d.page_content for d in docs])
    output = chain.run(job_description=job_desc, retrieved_entries=results)
    st.markdown("### Recommended SHL Assessments")
    st.markdown(output)
