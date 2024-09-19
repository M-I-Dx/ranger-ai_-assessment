import os
from question_answering.qa_system import RFXQuestionAnswering
from question_answering.chat_system import RFXConversationalQA


def run_qa_pipeline():
    """
    Run the question-answering system where the user can choose between conversational or non-conversational QA.
    """

    print("Choose the type of QA system you want to use:")
    print("1. Non-conversational (does not preserve conversation history)")
    print("2. Conversational (preserves conversation history)")
    
    qa_choice = input("Enter 1 or 2: ").strip()
    
    if qa_choice not in ['1', '2']:
        print("Invalid choice. Exiting.")
        return
    
    markdown_file_path = input("Enter the file path to the markdown file: ").strip()
    
    if not os.path.exists(markdown_file_path):
        print("The file path does not exist. Exiting.")
        return
    if qa_choice == '1':
        print("You have chosen the non-conversational QA system.")
        qa_system = RFXQuestionAnswering()
    else:
        print("You have chosen the conversational QA system.")
        qa_system = RFXConversationalQA()
    qa_system.load_document(markdown_file_path)

    print("You can now start asking questions. Type 'exit' to quit.")
    
    while True:
        question = input("Ask a question: ").strip()
        
        if question.lower() == 'exit':
            print("Exiting the QA system. Goodbye!")
            break
        try:
            answer = qa_system.ask_question(question)
            print(f"Answer: {answer}")
        except Exception as e:
            print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    run_qa_pipeline()