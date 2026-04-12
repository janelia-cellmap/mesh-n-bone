#include <CGAL/Simple_cartesian.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/boost/graph/IO/PLY.h>
#include <CGAL/Surface_mesh/IO/PLY.h>
#include <CGAL/boost/graph/generators.h>
#include <CGAL/IO/Color.h>

#include <iostream>
#include <fstream>
#include <cstring>



#include <CGAL/Polyhedron_3.h>
#include <CGAL/extract_mean_curvature_flow_skeleton.h>
#include <CGAL/boost/graph/split_graph_into_polylines.h>
#include <fstream>
#include <CGAL/subdivision_method_3.h>
#include <CGAL/Polygon_mesh_processing/repair.h>
#include <CGAL/Polygon_mesh_processing/border.h>


typedef CGAL::Simple_cartesian<double>                        Kernel;
typedef Kernel::Point_3                                       Point_3;
typedef CGAL::Surface_mesh<Point_3> Mesh;
typedef CGAL::Polyhedron_3<Kernel>                            Polyhedron;
typedef boost::graph_traits<Polyhedron>::vertex_descriptor    vertex_descriptor;
typedef CGAL::Mean_curvature_flow_skeletonization<Polyhedron> Skeletonization;
typedef Skeletonization::Skeleton                             Skeleton;
typedef Skeleton::vertex_descriptor                           Skeleton_vertex;
typedef Skeleton::edge_descriptor                             Skeleton_edge;
//only needed for the display of the skeleton as maximal polylines
struct Display_polylines{
  const Skeleton& skeleton;
  std::ofstream& out;
  int polyline_size;
  std::stringstream sstr;
  Display_polylines(const Skeleton& skeleton, std::ofstream& out)
    : skeleton(skeleton), out(out)
  {}
  void start_new_polyline(){
    polyline_size=0;
    sstr.str("");
    sstr.clear();
  }
  void add_node(Skeleton_vertex v){
    ++polyline_size;
    sstr << " " << v;//skeleton[v].point;
  }
  void end_polyline()
  {
    out << "p " << sstr.str() << "\n";
  }
};


void print_usage(const char* prog) {
  std::cerr
    << "Usage: " << prog << " <input.ply> <output.skel> [loop_subdivision_iterations] [--neuroglancer_format]\n"
    << "  <input.ply>                  : your triangulated mesh file\n"
    << "  <output.skel>                : where to write the skeleton\n"
    << "  [loop_subdivision_iterations]: optional integer (default 0)\n"
    << "  [--neuroglancer_format]      : flag to indicate input is in neuroglancer format\n";
}

// Function to read neuroglancer formatted mesh
bool read_neuroglancer_mesh(const std::string& filename, Polyhedron& mesh) {
  std::ifstream input(filename, std::ios::binary);
  if (!input) {
    std::cerr << "Error: cannot open file " << filename << std::endl;
    std::cerr << "Note: Make sure the file exists and you have read permissions" << std::endl;
    return false;
  }

  // Get file size for validation
  input.seekg(0, std::ios::end);
  std::streamsize file_size = input.tellg();
  input.seekg(0, std::ios::beg);

  if (file_size < 4) {
    std::cerr << "Error: file too small (" << file_size << " bytes), must be at least 4 bytes" << std::endl;
    return false;
  }

  // Check if this is a Draco-compressed file
  char magic[5] = {0};
  input.read(magic, 5);
  input.seekg(0, std::ios::beg);

  if (std::strncmp(magic, "DRACO", 5) == 0) {
    std::cerr << "Error: file appears to be Draco-compressed (starts with 'DRACO')" << std::endl;
    std::cerr << "This format is not supported. Please decompress the Draco file first." << std::endl;
    std::cerr << "You can use DracoPy in Python to decode it, or convert to PLY format." << std::endl;
    return false;
  }

  std::cerr << "Reading uncompressed neuroglancer mesh from " << filename << " (" << file_size << " bytes)" << std::endl;

  // Read number of vertices (4 bytes, little-endian uint32)
  uint32_t num_vertices;
  input.read(reinterpret_cast<char*>(&num_vertices), sizeof(uint32_t));
  if (!input) {
    std::cerr << "Error: failed to read num_vertices" << std::endl;
    return false;
  }

  // Sanity check: ensure num_vertices is reasonable given file size
  size_t min_expected_size = 4 + (num_vertices * 3 * sizeof(float)); // header + vertices only
  if (num_vertices > 1000000000 || min_expected_size > file_size * 10) {
    std::cerr << "Error: num_vertices value seems unreasonable: " << num_vertices << std::endl;
    std::cerr << "This may indicate the wrong file format or corrupted data" << std::endl;
    return false;
  }

  std::cerr << "Number of vertices: " << num_vertices << std::endl;

  // Read vertex positions (3 * num_vertices floats, little-endian)
  std::vector<Point_3> vertices;
  vertices.reserve(num_vertices);
  for (uint32_t i = 0; i < num_vertices; ++i) {
    float x, y, z;
    input.read(reinterpret_cast<char*>(&x), sizeof(float));
    input.read(reinterpret_cast<char*>(&y), sizeof(float));
    input.read(reinterpret_cast<char*>(&z), sizeof(float));
    if (!input) {
      std::cerr << "Error: failed to read vertex " << i << std::endl;
      return false;
    }
    vertices.push_back(Point_3(x, y, z));
  }

  // Calculate number of triangles from remaining bytes
  std::streampos current_pos = input.tellg();
  input.seekg(0, std::ios::end);
  std::streampos end_pos = input.tellg();
  size_t remaining_bytes = end_pos - current_pos;

  if (remaining_bytes % 12 != 0) {
    std::cerr << "Error: invalid number of remaining bytes for triangles: " << remaining_bytes << std::endl;
    return false;
  }

  uint32_t num_triangles = remaining_bytes / 12;
  input.seekg(current_pos);
  std::cerr << "Number of triangles: " << num_triangles << std::endl;

  // Read triangle indices (3 * num_triangles uint32s, little-endian)
  std::vector<std::array<uint32_t, 3>> faces;
  faces.reserve(num_triangles);
  for (uint32_t i = 0; i < num_triangles; ++i) {
    uint32_t a, b, c;
    input.read(reinterpret_cast<char*>(&a), sizeof(uint32_t));
    input.read(reinterpret_cast<char*>(&b), sizeof(uint32_t));
    input.read(reinterpret_cast<char*>(&c), sizeof(uint32_t));
    if (!input) {
      std::cerr << "Error: failed to read triangle " << i << std::endl;
      return false;
    }
    if (a >= num_vertices || b >= num_vertices || c >= num_vertices) {
      std::cerr << "Error: invalid vertex index in triangle " << i << std::endl;
      return false;
    }
    faces.push_back({a, b, c});
  }

  // Build polyhedron from vertices and faces
  typedef CGAL::Polyhedron_incremental_builder_3<Polyhedron::HalfedgeDS> Builder;
  struct Build_mesh : public CGAL::Modifier_base<Polyhedron::HalfedgeDS> {
    const std::vector<Point_3>& vertices;
    const std::vector<std::array<uint32_t, 3>>& faces;

    Build_mesh(const std::vector<Point_3>& v, const std::vector<std::array<uint32_t, 3>>& f)
      : vertices(v), faces(f) {}

    void operator()(Polyhedron::HalfedgeDS& hds) {
      Builder builder(hds, true);
      builder.begin_surface(vertices.size(), faces.size());

      // Add vertices
      for (const auto& v : vertices) {
        builder.add_vertex(v);
      }

      // Add faces
      for (const auto& f : faces) {
        builder.begin_facet();
        builder.add_vertex_to_facet(f[0]);
        builder.add_vertex_to_facet(f[1]);
        builder.add_vertex_to_facet(f[2]);
        builder.end_facet();
      }

      builder.end_surface();
    }
  };

  Build_mesh builder(vertices, faces);
  mesh.delegate(builder);

  std::cerr << "Successfully built polyhedron with " << mesh.size_of_vertices()
            << " vertices and " << mesh.size_of_facets() << " faces" << std::endl;

  return true;
}


// This example extracts a medially centered skeleton from a given mesh.

int main(int argc, char* argv[])
{
  if (argc < 3 || argc > 5) {
    print_usage(argv[0]);
    return EXIT_FAILURE;
  }

  // Check for --neuroglancer flag
  bool use_neuroglancer = false;
  for (int i = 3; i < argc; ++i) {
    if (std::strcmp(argv[i], "--neuroglancer_format") == 0) {
      use_neuroglancer = true;
      break;
    }
  }

  std::string comments;
  Polyhedron mesh;
  bool read_ok;

  // Load mesh based on format
  if (use_neuroglancer) {
    read_ok = read_neuroglancer_mesh(argv[1], mesh);
  } else {
    std::ifstream input(argv[1]);
    read_ok = CGAL::IO::read_PLY(input, mesh);
  }

  if(!read_ok){
   std::cout << "Error: reading the input file " << argv[1] << std::endl;
   return EXIT_FAILURE;
  }
  if (!CGAL::is_triangle_mesh(mesh))
  {
    std::cout << "Error: input geometry is not triangulated for " << argv[1] << std::endl;
    return EXIT_FAILURE;
  }

  // parse optional loop subdivision count
  int loop_subdivision_iterations = 0;
  if (argc >= 4) {
    for (int i = 3; i < argc; ++i) {
      if (std::strcmp(argv[i], "--neuroglancer_format") != 0) {
        try {
          loop_subdivision_iterations = std::stoi(argv[i]);
          if (loop_subdivision_iterations < 0) throw std::out_of_range("negative");
          break;
        }
        catch (const std::invalid_argument&) {
          std::cerr << "Error: loop_subdivision_iterations must be an integer\n";
          print_usage(argv[0]);
          return EXIT_FAILURE;
        }
        catch (const std::out_of_range&) {
          std::cerr << "Error: loop_subdivision_iterations out of range\n";
          print_usage(argv[0]);
          return EXIT_FAILURE;
        }
      }
    }
  }

  CGAL_assertion(mesh.is_valid());
  // Note: is_connected() and has_border_edges() not available in CGAL 6.0.1
  // CGAL_assertion( CGAL::Polygon_mesh_processing::is_connected(mesh) );
  // CGAL_assertion( !CGAL::Polygon_mesh_processing::has_border_edges(mesh) );
  if (loop_subdivision_iterations > 0){
    CGAL::Subdivision_method_3::Loop_subdivision(mesh, CGAL::parameters::number_of_iterations(loop_subdivision_iterations));
  }

  Skeleton skeleton;

  // Create MCF skeletonization object with explicit parameters
  typedef CGAL::Mean_curvature_flow_skeletonization<Polyhedron> MCF_Skel;
  MCF_Skel mcs(mesh);

  // Tweak parameters to try to keep the skeleton inside
  mcs.set_is_medially_centered(true);              // redundant with defaults, but explicit
  mcs.set_medially_centered_speed_tradeoff(0.3);   // > default .3 worked well
  mcs.set_quality_speed_tradeoff(0.5);             // > default .5 worked well

  mcs.contract_until_convergence();
  mcs.convert_to_skeleton(skeleton);
  
  // Output all the edges of the skeleton.
  if (boost::num_vertices(skeleton) == 0)
  {
    std::cout << "Error: the skeleton is too small for " << argv[1] << std::endl;
    return EXIT_FAILURE;
  }


  //  Output skeleton points and the corresponding surface points
  std::ofstream output(argv[2]);
  // assign indices to vertices
  for(Skeleton_vertex v : CGAL::make_range(vertices(skeleton))){
    Point_3 skeleton_vertex = skeleton[v].point;
    float average_radius=0, radius = 0;
    if (skeleton[v].vertices.size() == 0)
      {// some vertices aren't associated with mesh vertices but we still want to calculate a radius for them
        float min_radius = std::numeric_limits<float>::max();
        //loop over all vertices of mesh and find the minimum distance between the skeleton vertex and the mesh vertices
        for(vertex_descriptor mesh_vd : vertices(mesh)){
          Point_3 mesh_vertex = get(CGAL::vertex_point, mesh, mesh_vd);
          radius = std::sqrt(CGAL::squared_distance(mesh_vertex, skeleton_vertex));
          if(radius < min_radius)
            min_radius = radius;
        }
        average_radius = min_radius;
      }
      else{
        // get the average associated vertex distance
        for(vertex_descriptor vd : skeleton[v].vertices){
          Point_3 associated_mesh_vertex = get(CGAL::vertex_point, mesh, vd);
          average_radius += std::sqrt(CGAL::squared_distance(associated_mesh_vertex, skeleton_vertex));
        }
        average_radius /= skeleton[v].vertices.size();
      }
      // use a higher precision after the decimal point for the output
      output << "v " << std::fixed << std::setprecision(8) << skeleton[v].point << " " << average_radius << "\n" ;
      //output << "v " << std::fixed << std::setprecision(std::numeric_limits<double>::digits10) << skeleton[v].point << " " << average_radius << "\n" ;

  }

  for(Skeleton_edge e : CGAL::make_range(edges(skeleton)))
  {
    Skeleton_vertex source_vertex = source(e, skeleton);
    Skeleton_vertex target_vertex = target(e, skeleton);

    output << "e " << source_vertex << " " << target_vertex << "\n";

  }


  //output.close();

  // Output all the edges of the skeleton.
  //std::ofstream output("skel-poly.polylines.txt");
  Display_polylines display(skeleton,output);
  CGAL::split_graph_into_polylines(skeleton, display);
  output.close();

  // //  Output skeleton points and the corresponding surface points
  // output.open("correspondance-poly.polylines.txt");
  // for(Skeleton_vertex v : CGAL::make_range(vertices(skeleton)))
  //   for(vertex_descriptor vd : skeleton[v].vertices)
  //     output << "2 " << skeleton[v].point << " "
  //                    << get(CGAL::vertex_point, mesh, vd)  << "\n";

  // std::cout << "Number of vertices of the skeleton: " << boost::num_vertices(skeleton) << "\n";
  // std::cout << "Number of edges of the skeleton: " << boost::num_edges(skeleton) << "\n";


  //std::cout << "Success in writing the skeleton to " << argv[2] << std::endl;
  return EXIT_SUCCESS;

}
