{#
  Rainbow collision kernel body, with a CLUSTER level between each link's
  bounding sphere and its spheres. `clusters` (from the configuration JSON,
  indexed like link_names) gives each link's spheres in spatial clusters; a
  cluster is screened by a ball centred on one of its own spheres (`anchor`,
  whose world position the FK already computed) with a constant radius
  covering every member. Same verdicts as the flat check (a screen only ever
  skips sphere tests that cannot hit), far fewer tests for a 1000-sphere model.
  Sphere tests (environment and self) run in two small out-of-line loops
  (sphere_set_environment, sphere_sets_collide) over constant index tables, so
  the code size is O(clusters + cluster pairs), not O(spheres + sphere pairs),
  and the kernel function stays a loop-free chain of screens (clang compiles a
  function holding thousands of early-exit loops super-linearly slowly).
#}
{% for i in range(length(links_with_geometry)) %}
{% set array_index = length(links_with_geometry) - i - 1 %}
{% set link_index = at(links_with_geometry, array_index) %}
{% set link_clusters = at(clusters, link_index) %}
{% set bs_loc = (n_spheres + array_index) * 4 %}

// environment vs. {{ at(link_names, link_index) }}
if (sphere_environment_in_collision(environment, y[{{bs_loc + 0}}], y[{{bs_loc + 1}}], y[{{bs_loc + 2}}], y[{{bs_loc + 3}}]))
{
{% for c in link_clusters %}
{% if length(link_clusters) > 1 %}
    if (sphere_environment_in_collision(environment, y[{{ c.anchor * 4 + 0 }}], y[{{ c.anchor * 4 + 1 }}], y[{{ c.anchor * 4 + 2 }}], {{ c.radius }}F))
{% endif %}
    {
        static constexpr std::array<std::size_t, {{ length(c.spheres) }}> se{ {% for s in c.spheres %}{{ s * 4 }}{% if not loop.is_last %}, {% endif %}{% endfor %} };
        if (sphere_set_environment(environment, y, se.data(), se.size())) { return false; }
    }
{% endfor %}
}
{% endfor %}

//
// robot self-collisions
//
{% for i in range(length(allowed_link_pairs)) %}
{% set pair = at(allowed_link_pairs, i) %}
{% set link_1_index = at(pair, 0) %}
{% set link_2_index = at(pair, 1) %}
{% set link_1_bs_loc = (n_spheres + at(bounding_sphere_index, link_1_index)) * 4 %}
{% set link_2_bs_loc = (n_spheres + at(bounding_sphere_index, link_2_index)) * 4 %}
{% set clusters_1 = at(clusters, link_1_index) %}
{% set clusters_2 = at(clusters, link_2_index) %}

// {{ at(link_names, link_1_index) }} vs. {{ at(link_names, link_2_index) }}
if (sphere_sphere_self_collision<decltype(x[0])>(y[{{link_1_bs_loc + 0}}], y[{{link_1_bs_loc + 1}}], y[{{link_1_bs_loc + 2}}], y[{{link_1_bs_loc + 3}}],
                                                 y[{{link_2_bs_loc + 0}}], y[{{link_2_bs_loc + 1}}], y[{{link_2_bs_loc + 2}}], y[{{link_2_bs_loc + 3}}]))
{
{% for c1 in clusters_1 %}
{% for c2 in clusters_2 %}
{% if length(clusters_1) > 1 or length(clusters_2) > 1 %}
    if (sphere_sphere_self_collision<decltype(x[0])>(y[{{ c1.anchor * 4 + 0 }}], y[{{ c1.anchor * 4 + 1 }}], y[{{ c1.anchor * 4 + 2 }}], {{ c1.radius }}F,
                                                     y[{{ c2.anchor * 4 + 0 }}], y[{{ c2.anchor * 4 + 1 }}], y[{{ c2.anchor * 4 + 2 }}], {{ c2.radius }}F))
{% endif %}
    {
        static constexpr std::array<std::size_t, {{ length(c1.spheres) }}> s1{ {% for s in c1.spheres %}{{ s * 4 }}{% if not loop.is_last %}, {% endif %}{% endfor %} };
        static constexpr std::array<std::size_t, {{ length(c2.spheres) }}> s2{ {% for s in c2.spheres %}{{ s * 4 }}{% if not loop.is_last %}, {% endif %}{% endfor %} };
        if (sphere_sets_collide(y, s1.data(), s1.size(), s2.data(), s2.size())) { return false; }
    }
{% endfor %}
{% endfor %}
}
{% endfor %}
