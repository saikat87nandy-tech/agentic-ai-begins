from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA

load_dotenv()

def main():
    print('Hi From Agenti AI')
    print('Processing...')
    information = """Acknowledgement Number:489049280280726 Date of filing : 28-Jul-2026
    INDIAN INCOME TAX RETURN ACKNOWLEDGEMENT
    [Where the data of the Return of Income in Form ITR-1(SAHAJ), ITR-2, ITR-3, ITR-4(SUGAM), ITR-5, ITR-6, ITR-7
    filed and verified]
    (Please see Rule 12 of the Income-tax Rules, 1962)
    Assessment
    Year
    2026-27
    PAN AKZPN4701E
    Name SAIKAT NANDY
    Address PURBASA, GREEN PARK, Baksara, HOWRAH, Baksara S.O, 32-West Bengal, 91-INDIA, 711110
    Status Individual Form Number ITR-1
    Filed u/s 139(1)-On or before due date e-Filing Acknowledgement Number 489049280280726
    Current Year business loss, if any 1 0
    Total Income 1A 10,97,530
    Book Profit under MAT, where applicable 2 0
    Adjusted Total Income under AMT, where applicable 3 0
    Net tax payable 4 0
    Interest and Fee Payable 5 0
    Total tax, interest and Fee payable 6 0
    Taxes Paid 7 24,787
    (+) Tax Payable /(-) Refundable (6-7) 8 (-) 24,790
    Accreted Income as per section 115TD 9 0
    Additional Tax payable u/s 115TD 10 0
    Interest payable u/s 115TE 11 0
    Additional Tax and interest payable 12 0
    Tax and interest paid 13 0
    (+) Tax Payable /(-) Refundable (12-13) 14 0
    Income Tax Return electronically transmitted on 28-Jul-2026 10:01:32 from IP address 116.193.140.180
    and verified by SAIKAT NANDY having PAN AKZPN4701E on 28-Jul-2026 using
    paper ITR-Verification Form /Electronic Verification Code 8UZXAPKKRI generated through Aadhaar OTP
    mode
    System Generated 
    Barcode/QR Code
    AKZPN4701E014890492802807264fcf42dcd1a64462fe368034cac6fcea75575b22
     DO NOT SEND THIS ACKNOWLEDGEMENT TO CPC, BENGALURU  
    Taxable Income and Tax Details Accreted Income and Tax Detail"""

    summary_template="""You are expert document analyst. Carefully read the following text:
    {information}
    Provide the following:
    1. Summary: A concise 3-4 line overview
    2. Key Points: Most important information in bullet point
    3. Critical Figure/Dates: Any important numbers, amount or dates 
    """

    summary_prompt_template = PromptTemplate(
        input_variables= ["information"],
        template=summary_template
    )

    llm=ChatNVIDIA(temperature = 0, model="nvidia/nemotron-3-ultra-550b-a55b")

    Chain=summary_prompt_template | llm
    response = Chain.invoke(
        input={"information": information}
    )

    print(response.content)

main()
