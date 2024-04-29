#include<iostream>
#include<fstream>
#include<vector>
#include<string>
#include<sstream>
#include<cmath>

using namespace std;

void export_vtu(const std::string &file, vector<vector<double>> x, vector<vector<int>> element, vector<vector<double>> v, vector<double> pressure, vector<double> c, vector<int> index)
{
  FILE *fp;
  if ((fp = fopen(file.c_str(), "w")) == NULL)
  {
    cout << file << " open error" << endl;
    exit(1);
  }

  fprintf(fp, "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt32\">\n");
  fprintf(fp, "<UnstructuredGrid>\n");
  fprintf(fp, "<Piece NumberOfPoints= \"%d\" NumberOfCells= \"%d\" >\n", x.size(), element.size());
  fprintf(fp, "<Points>\n");
  int offset = 0;
  fprintf(fp, "<DataArray type=\"Float64\" Name=\"Position\" NumberOfComponents=\"3\" format=\"appended\" offset=\"%d\"/>\n",offset);
  offset += sizeof(int) + sizeof(double) * x.size() * 3;
  fprintf(fp, "</Points>\n");

  fprintf(fp, "<Cells>\n");
  fprintf(fp, "<DataArray type=\"Int64\" Name=\"connectivity\" format=\"ascii\">\n");
  for (int i = 0; i < element.size(); i++){
    for (int j = 0; j < element[i].size(); j++) fprintf(fp, "%d ", element[i][j]);
    fprintf(fp, "\n");
  }
  fprintf(fp, "</DataArray>\n");
  fprintf(fp, "<DataArray type=\"Int64\" Name=\"offsets\" format=\"ascii\">\n");
  int num = 0;
  for (int i = 0; i < element.size(); i++)
  {
    num += element[i].size();
    fprintf(fp, "%d\n", num);
  }
  fprintf(fp, "</DataArray>\n");
  fprintf(fp, "<DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n");
  for (int i = 0; i < element.size(); i++){
    if(element[i].size()==4) fprintf(fp, "%d\n", 10);
    else if(element[i].size()==3) fprintf(fp, "%d\n", 5);
    else if(element[i].size()==6) fprintf(fp, "%d\n", 13);
    else if(element[i].size()==5) fprintf(fp, "%d\n", 14);
    else if(element[i].size()==8) fprintf(fp, "%d\n", 42);
  }
  fprintf(fp, "</DataArray>\n");
  fprintf(fp, "</Cells>\n");

  fprintf(fp, "<PointData>\n");
  fprintf(fp, "<DataArray type=\"Float64\" Name=\"velocity[m/s]\" NumberOfComponents=\"3\" format=\"appended\" offset=\"%d\"/>\n",offset);
  offset += sizeof(int) + sizeof(double) * x.size() * 3;

  fprintf(fp, "<DataArray type=\"Float64\" Name=\"pressure[Pa]\" NumberOfComponents=\"1\" format=\"appended\" offset=\"%d\"/>\n",offset);
  offset += sizeof(int) + sizeof(double) * x.size();

  fprintf(fp, "<DataArray type=\"Float64\" Name=\"concentration[-]\" NumberOfComponents=\"1\" format=\"appended\" offset=\"%d\"/>\n",offset);
  offset += sizeof(int) + sizeof(double) * x.size();
  fprintf(fp, "</PointData>\n");

  fprintf(fp, "<CellData>\n");
  fprintf(fp, "</CellData>\n");
  fprintf(fp, "</Piece>\n");
  fprintf(fp, "</UnstructuredGrid>\n");
  fprintf(fp, "<AppendedData encoding=\"raw\">\n");
  fprintf(fp, "_");
  fclose(fp);

  fstream ofs;
  ofs.open(file.c_str(), ios::out | ios::app | ios_base::binary);
  double *data_d = new double[x.size()*3];
  num = 0;
  int size=0;
  for (int ic = 0; ic < x.size(); ic++){
    for(int j=0;j<3;j++){
      data_d[num] = x[ic][j];
      num++;
    }
  }

  size=sizeof(double)*x.size()*3;
  ofs.write((char *)&size, sizeof(size));
  ofs.write((char *)data_d, size);

  num=0;
  for (int ic = 0; ic < x.size(); ic++){
      data_d[num]   = v[ic][0];
      data_d[num+1] = v[ic][1];
      data_d[num+2] = v[ic][2];
      num=num+3;
  }

  size=sizeof(double)*x.size()*3;
  ofs.write((char *)&size, sizeof(size));
  ofs.write((char *)data_d, size);

  num=0;
  for (int ic = 0; ic < x.size(); ic++){
      data_d[num]   = pressure[ic];
      num++;
  }
  size=sizeof(double)*x.size();
  ofs.write((char *)&size, sizeof(size));
  ofs.write((char *)data_d, size);
  num=0;
  for (int ic = 0; ic < x.size(); ic++){
      data_d[num]   = c[ic];
      //cout << ic << " " << c.size() << " " << c[ic] << endl;
      num++;
  }
  size=sizeof(double)*x.size();
  ofs.write((char *)&size, sizeof(size));
  ofs.write((char *)data_d, size);

  delete data_d;
  ofs.close();

  if ((fp = fopen(file.c_str(), "a")) == NULL)
  {
    cout << file << " open error" << endl;
    exit(1);
  }
  fprintf(fp, "\n</AppendedData>\n");
  fprintf(fp, "</VTKFile>\n");
  fclose(fp);
}

int CountNumbersOfTextLines(const string &filePath)
{
  long i = 0;

  ifstream ifs( filePath );

  if( ifs )
  {
    string line;

    while( true )
    {
      getline( ifs, line );
      i++;
      if( ifs.eof() )
        break;
    }
  }
  return i-1;
}

void read_geometry_node(std::vector<std::vector<double>> &x,int &numOfNode,const string &file)
{
  string str,tmp;
  numOfNode = CountNumbersOfTextLines(file);

  x.resize(numOfNode);
  for(int i=0; i<x.size(); i++){
    x[i].resize(3);
  }

  ifstream node_file(file);
  if(!node_file){
    cout << "Error:Input "<< file << " not found" << endl;
    exit(1);
  }

  for(int i=0;i<numOfNode;i++){

    getline(node_file,str);
    istringstream stream(str);

    for(int j=0;j<3;j++){
      getline(stream,tmp,',');
      x[i][j] = stof(tmp);
    }
  }
}

void read_elementType(std::vector<int> &elementType,const int &numOfElm, const string &file)
{
  string str,tmp;
  ifstream element_file(file);
  if(!element_file){
    cout << "Error:Input "<< file << " not found" << endl;
    exit(1);
  }

  while(getline(element_file, str)){
    elementType.push_back(stoi(str));
  }
}

void read_geometry_element(std::vector<vector<int>> &element, const int &numOfElm, const string &file)
{
  string str;
  ifstream ifs(file);
  for(int i=0; i<numOfElm; i++){
    getline(ifs, str);
    istringstream ss(str);
    for(int j=0; j<element[i].size(); j++){
      getline(ss, str, ',');
      element[i][j] = stoi(str);
    }
  }
}

vector<double> read_scalar_value(string filename, vector<double> &c, vector<int> index)
{
    ifstream ifs(filename);
    if(!ifs){
      cout << filename + " not found !" << endl;
      exit(1);
    }
    string str;
    getline(ifs, str);
    int count = 0;
    while(getline(ifs,str)){
        istringstream stream(str);
        for(int i=0; i<2; i++){
            getline(stream,str,',');
            if(i==1) c[count] = (stod(str));
        }
        count++;
    }
    return c;
}

vector<vector<double>> read_reference_velocity(string filename)
{
    ifstream ifs(filename);
    if(!ifs){
      cout << filename + " not found !" << endl;
      exit(1);
    }
    string str;
    vector<vector<double>> velocity;
    getline(ifs, str);
    while(getline(ifs,str)){
        istringstream stream(str);
        vector<double> tmp_vel;
        for(int i=0; i<3; i++){
            getline(stream,str,',');
            tmp_vel.push_back(stod(str));
        }
        velocity.push_back(tmp_vel);
    }
    return velocity;
} 

int main()
{
    string base_dir = "/mnt/d//test_case_pulsatile_csv";
    string x_file = base_dir + "/node.csv";
    string element_file = base_dir + "/element.csv";
    string elementType_file = base_dir + "/elementType.csv";

    vector<vector<double>> x;
    vector<vector<int>> element;
    vector<int> elementType;
    vector<int> index;

    int numOfNode = CountNumbersOfTextLines(x_file);
    int numOfElm = CountNumbersOfTextLines(element_file);

    read_elementType(elementType,numOfElm, elementType_file);
    element.resize(numOfElm);
    for(int i=0; i<numOfElm; i++){
      if(elementType[i]==10) element[i].resize(4);
      if(elementType[i]==5) element[i].resize(3);
      else if(elementType[i]==14) element[i].resize(5);
      else if(elementType[i]==13) element[i].resize(6);
      else if(elementType[i]==42) element[i].resize(8);
    }
    cout << "read element" << endl;
    read_geometry_element(element, numOfElm, element_file);
    cout << "read node" << endl;
    read_geometry_node(x, numOfNode, x_file);

    cout << "read scalar" << endl;
    string pred_file = "test_pred_c.csv";
    vector<double> c(numOfNode);
    read_scalar_value(pred_file, c, index);
    pred_file = "test_pred_p.csv";
    vector<double> p(numOfNode);
    read_scalar_value(pred_file, p, index);
    cout << "read pressure" << endl;
    pred_file = "test_pred_u.csv";
    vector<double> u(numOfNode);
    read_scalar_value(pred_file, u, index);
    cout << "read u" << endl;
    pred_file = "test_pred_v.csv";
    vector<double> v(numOfNode);
    read_scalar_value(pred_file, v, index);
    cout << "read w" << endl;
    pred_file = "test_pred_w.csv";
    vector<double> w(numOfNode);
    read_scalar_value(pred_file, w, index);

    cout << "append velocity" << endl;
    vector<vector<double>> velocity;
    velocity.push_back(u);
    velocity.push_back(v);
    velocity.push_back(w);

    string file_reference_velocity = "/mnt/d//test_case_pulsatile_csv/reference_velocity.csv";
    vector<vector<double>> reference_velocity = read_reference_velocity(file_reference_velocity);

    vector<vector<double>> error_vel;
    for(int i=0; i<x.size(); i++){
        vector<double> tmp_velocity;
        // u error
        double error_u = fabs(velocity[0][i] * 0.05 - reference_velocity[i][0]);
        double error_v = fabs(velocity[1][i] * 0.05 - reference_velocity[i][1]);
        double error_w = fabs(velocity[2][i] * 0.05 - reference_velocity[i][2]);
        tmp_velocity.push_back(error_u);
        tmp_velocity.push_back(error_v);
        tmp_velocity.push_back(error_w);
        error_vel.push_back(tmp_velocity);
    }
    string output = "error.vtu";
    export_vtu(output, x, element, error_vel, p, c, index);
}