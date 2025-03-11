import streamlit as st
from crewai_tools import ScrapeWebsiteTool
import ollama
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Function to scrape website content with error handling
def scrape_website(url):
    try:
        tool = ScrapeWebsiteTool(website_url=url)
        scraped_data = tool.run()
        
        # Validate the scraped data
        if not scraped_data or not isinstance(scraped_data, str):
            raise ValueError("Invalid or empty response from the website.")
        
        return scraped_data
    except Exception as e:
        st.error(f"An error occurred while scraping the website: {e}")
        return None

# Function to clean the text
def clean_text(text):
    # Remove null bytes and strip whitespace
    text = text.replace("\x00", "").strip()
    return text

# Function to generate a summary in bullet points
def generate_summary(text):
    prompt = f"Provide a summary of the following text in bullet points:\n\n{text}\n\nSummary:"
    try:
        response = ollama.chat(model="llama3.2:1b", messages=[{"role": "user", "content": prompt}])
        summary = response['message']['content']
        # Split the summary into lines
        lines = summary.split("\n")
        # Filter out empty lines and ensure each non-empty line starts with a bullet
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line:  # Only process non-empty lines
                if not line.startswith("- "):
                    line = "- " + line  # Add bullet if not present
                cleaned_lines.append(line)
        # Join the lines back together
        return "\n".join(cleaned_lines)
    except Exception as e:
        st.error(f"Error generating summary: {e}")
        logging.error(f"Summary generation error: {e}", exc_info=True)
        return "Failed to generate a summary."

# Function to answer a user question
def answer_question(question, text):
    # Generate an answer using Llama3.2:1b directly with the scraped text
    prompt = f"Answer the following question based on the text:\n\n{text}\n\nQ: {question}\nA:"
    try:
        response = ollama.chat(model="llama3.2:1b", messages=[{"role": "user", "content": prompt}])
        return response['message']['content']
    except Exception as e:
        st.error(f"Error generating answer: {e}")
        logging.error(f"Answer generation error: {e}", exc_info=True)
        return "Failed to generate an answer."

# Streamlit App
def main():
    st.title("Website Scraper with Summary and Q&A")
    
    # Use session state to store the scraped text and summary
    if "scraped_text" not in st.session_state:
        st.session_state["scraped_text"] = None
    if "summary" not in st.session_state:
        st.session_state["summary"] = None

    # Input field for the website URL
    url = st.text_input("Enter the website URL to scrape:")
    
    if st.button("Scrape Website"):
        if url:
            # Scrape the website
            text = scrape_website(url)
            
            if text:
                st.subheader("Scraped Content:")
                st.write(text[:1000] + "...")  # Display the first 1000 characters
                
                # Clean the text
                cleaned_text = clean_text(text)
                
                if not cleaned_text:
                    st.error("Cleaned text is empty. Nothing to process.")
                    return
                
                # Store the cleaned text in session state
                st.session_state["scraped_text"] = cleaned_text
                logging.debug(f"Scraped Content: {cleaned_text[:500]}...")  # Log first 500 chars
                
                # Generate and store the summary
                summary = generate_summary(cleaned_text)
                st.session_state["summary"] = summary
                
                st.subheader("Website Summary:")
                st.markdown(summary)  # Display summary as markdown for bullet points
                
                st.success("Website content successfully scraped, summarized, and stored.")
        else:
            st.warning("Please enter a valid website URL.")
    
    # Allow the user to ask questions
    st.subheader("Ask a Question")
    user_question = st.text_input("Enter your question:")
    
    if st.button("Get Answer"):
        if user_question:
            if st.session_state["scraped_text"]:
                answer = answer_question(user_question, st.session_state["scraped_text"])
                st.subheader("Answer:")
                st.write(answer)
            else:
                st.warning("No website content has been scraped yet. Please scrape a website first.")
        else:
            st.warning("Please enter a question.")

if __name__ == "__main__":
    main()