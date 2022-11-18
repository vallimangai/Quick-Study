import PyPDF2
def pdfTotext(n):
    pdfFileObj = open(n, 'rb')
    pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
    num_of_pages=pdfReader.numPages
    output = ''
    for i in range(num_of_pages):
        page = pdfReader.getPage(i)
        output+=page.extractText()
    pdfFileObj.close()
    return output
# pdfTotext('1.pdf')