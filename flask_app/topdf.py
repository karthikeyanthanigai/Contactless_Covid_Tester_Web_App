from fpdf import FPDF 


# save FPDF() class into a 
# variable pdf 
pdf = FPDF() 

# Add a page 
pdf.add_page() 

# set style and size of font 
# that you want in the pdf 
pdf.set_font("Arial", size = 30) 

# create a cell 
pdf.cell(200, 10, txt = "VirTest", 
		ln = 1, align = 'L') 

# add another cell 
pdf.cell(200, 10, txt = "www.virtest.com", 
		ln = 2, align = 'L') 
pdf.cell(200, 10, txt = "", 
		ln = 2, align = 'L') 
pdf.cell(200, 10, txt = "Dr.Jason Wick                  Date:", 
		ln = 2, align = 'L')
pdf.cell(250, 10, txt = "-----------------------------------------------------", 
		ln = 2, align = 'L')
pdf.cell(200, 10, txt = "", 
		ln = 2, align = 'L') 




pdf.cell(200, 10, txt = "Name:                        Age:", 
		ln = 2, align = 'L')
pdf.cell(200, 10, txt = "Heart BPM:                Respiratory Rate:", 
		ln = 2, align = 'L')

pdf.cell(250, 10, txt = "Cough Severity:", 
		ln = 2, align = 'L')
pdf.cell(250, 10, txt = "-----------------------------------------------------", 
		ln = 2, align = 'L')
pdf.cell(250, 10, txt = "Remarks:", 
		ln = 2, align = 'L')



# save the pdf with name .pdf 
pdf.output("VirtTest_Prescription.pdf") 
